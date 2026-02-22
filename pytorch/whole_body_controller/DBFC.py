"""
WBC (Whole Body Controller) — C++ DBFC_core.cpp 대응.

두 가지 모드:
  (A) IK + PD: WholeBodyIK → PD 토크 (위치 제어)
  (B) DBFC:    BalanceTask → CoM Dynamics → ForceOptimizer → WholeBodyTorque
               (힘 제어, 논문 식 19-20)

배치 PyTorch 구현. Isaac Lab API 기반.
"""

import torch
from pytorch.config.g1_config import WalkingConfig
from pytorch.whole_body_controller.whole_body_ik import WholeBodyIK
from pytorch.whole_body_controller.tasks.balance_task import BalanceTask
from pytorch.dynamics_model.com_dynamics import CenterOfMassDynamics
from pytorch.whole_body_controller.Force_Optimizier import ForceOptimizer
from pytorch.whole_body_controller.whole_body_torque import WholeBodyTorqueGenerator
from pytorch.constraints.friction_cone import BatchedFrictionCone
from pytorch.constraints.cop_limits import BatchedCoPLimits


class WBC:
    """Whole Body Controller — C++ WBC 대응.

    IK 모드와 DBFC 모드를 모두 지원.
    """

    def __init__(self, cfg: WalkingConfig, nv: int = 29, na: int = 23, nc: int = 12):
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.nv = nv
        self.na = na
        self.nc = nc

        # 서브모듈
        self.ik = WholeBodyIK(cfg, nv=nv, na=na)
        self.balance_task = BalanceTask(cfg)
        self.com_dynamics = CenterOfMassDynamics(cfg)
        self.force_opt = ForceOptimizer(cfg, num_vars=nc, num_ineq=24)
        self.torque_gen = WholeBodyTorqueGenerator(cfg, nv=nv, na=na, nc=nc)

        # 제약조건
        self.friction_cone = BatchedFrictionCone(cfg)
        # CoP limits — G1 발 크기 기준 (추후 config로 이동 가능)
        self.cop_limits = BatchedCoPLimits(
            cfg, dX_max=0.05, dX_min=-0.05, dY_max=0.02, dY_min=-0.02
        )

    def compute_ik(
        self,
        jacobians: torch.Tensor,     # (B, n_bodies, 6, nv)
        q_curr: torch.Tensor,        # (B, nq)
        dq_curr: torch.Tensor,       # (B, nv)
        com_jac: torch.Tensor,       # (B, 3, nv)
        com_pos: torch.Tensor,       # (B, 3)
        rf_body_idx: int,
        lf_body_idx: int,
        com_des: torch.Tensor,       # (B, 3)
        rf_pos_des: torch.Tensor,    # (B, 3)
        lf_pos_des: torch.Tensor,    # (B, 3)
        rf_pos_curr: torch.Tensor,   # (B, 3)
        lf_pos_curr: torch.Tensor,   # (B, 3)
        rf_ori_curr: torch.Tensor,   # (B, 3, 3)
        lf_ori_curr: torch.Tensor,   # (B, 3, 3)
        rf_vel_ff: torch.Tensor = None,
        lf_vel_ff: torch.Tensor = None,
        rf_ori_des: torch.Tensor = None,    # (B, 3, 3) — 발 목표 자세 (None=현재 유지)
        lf_ori_des: torch.Tensor = None,
        # feedforward 토크용 추가 인자
        mass_matrix: torch.Tensor = None,   # (B, nv, nv)
        nle: torch.Tensor = None,           # (B, nv)
        com_vel: torch.Tensor = None,       # (B, 3)
        contact_state: torch.Tensor = None, # (B, 2)
        mpc_ddx: float = 0.0,               # MPC x 가속도 (walking phase)
        mpc_ddy: float = 0.0,               # MPC y 가속도 (walking phase)
    ) -> torch.Tensor:
        """IK(fb) + ForceOpt+TorqueGen(ff) — C++ standingLoop/wbcLoop 대응.

        tau_fb: IK → PD 토크 (피드백)
        tau_ff: BalanceTask → ForceOpt → TorqueGen (피드포워드)
        Returns:
            tau: (B, na) = tau_ff + tau_fb
        """
        # ── τ_fb: IK + PD ──
        self.ik.compute(
            jacobians=jacobians,
            q_curr=q_curr,
            dq_curr=dq_curr,
            com_jac=com_jac,
            com_pos=com_pos,
            rf_body_idx=rf_body_idx,
            lf_body_idx=lf_body_idx,
            com_des=com_des,
            rf_pos_des=rf_pos_des,
            lf_pos_des=lf_pos_des,
            rf_pos_curr=rf_pos_curr,
            lf_pos_curr=lf_pos_curr,
            rf_ori_curr=rf_ori_curr,
            lf_ori_curr=lf_ori_curr,
            rf_vel_ff=rf_vel_ff,
            lf_vel_ff=lf_vel_ff,
            rf_ori_des=rf_ori_des,
            lf_ori_des=lf_ori_des,
        )
        tau_fb = self.ik.compute_pd_torque(q_curr, dq_curr)

        # ── τ_ff: BalanceTask → ForceOpt → TorqueGen (C++ standingLoop/wbcLoop 대응) ──
        if mass_matrix is not None and nle is not None and com_vel is not None:
            B = com_pos.shape[0]
            device = self.device
            dtype = self.dtype
            m = self.cfg.robot_mass
            g = self.cfg.gravity

            # BalanceTask: CoM PD 보정 가속도 (C++ balance_task_.update 대응)
            com_dot_des = torch.zeros_like(com_des)
            ddc_pd = self.balance_task.update(com_pos, com_vel, com_des, com_dot_des)
            # C++: z축은 PD 미적용
            ddc_pd[:, 2] = 0.0

            # C++ 방식 K 행렬: 단순 힘 매핑 (6×12)
            # K(0,0)=1 K(1,1)=1 K(2,2)=1  (오른발 force)
            # K(0,6)=1 K(1,7)=1 K(2,8)=1  (왼발 force)
            K = torch.zeros(B, 6, 12, device=device, dtype=dtype)
            K[:, 0, 0] = 1.0; K[:, 1, 1] = 1.0; K[:, 2, 2] = 1.0
            K[:, 0, 6] = 1.0; K[:, 1, 7] = 1.0; K[:, 2, 8] = 1.0

            # C++ 방식 u_des: [m*(ddx_mpc + ddc_pd_x), m*(ddy_mpc + ddc_pd_y), m*g, 0, 0, 0]
            u_vec = torch.zeros(B, 6, device=device, dtype=dtype)
            u_vec[:, 0] = m * (mpc_ddx + ddc_pd[:, 0])
            u_vec[:, 1] = m * (mpc_ddy + ddc_pd[:, 1])
            u_vec[:, 2] = m * g
            # u_vec[:, 3:6] = 0  (모멘트 없음)

            # 제약조건
            A_fric_r, l_fric_r, u_fric_r = self.friction_cone.update()
            A_fric_l, l_fric_l, u_fric_l = self.friction_cone.update()
            A_fric = torch.zeros(B, 10, 12, device=device, dtype=dtype)
            A_fric[:, :5, :3] = A_fric_r
            A_fric[:, 5:, 6:9] = A_fric_l
            l_fric = torch.cat([l_fric_r, l_fric_l], dim=-1)
            u_fric = torch.cat([u_fric_r, u_fric_l], dim=-1)

            if contact_state is None:
                contact_state = torch.ones(B, 2, device=device, dtype=dtype)
            A_cop, l_cop, u_cop = self.cop_limits.update(contact_state)

            A_ineq = torch.cat([A_fric, A_cop], dim=1)
            l_ineq = torch.cat([l_fric, l_cop], dim=1)
            u_ineq = torch.cat([u_fric, u_cop], dim=1)

            # 스윙 발 제약: 스윙 발의 6개 변수를 0으로 강제 (C++ 대응)
            A_swing = torch.zeros(B, 6, 12, device=device, dtype=dtype)
            l_swing = torch.full((B, 6), -1e6, device=device, dtype=dtype)
            u_swing = torch.full((B, 6), 1e6, device=device, dtype=dtype)

            rf_swing = (contact_state[:, 0] < 0.5)  # (B,)
            lf_swing = (contact_state[:, 1] < 0.5)  # (B,)

            for j in range(6):
                A_swing[:, j, j] = 1.0  # 기본: 오른발 열 (dummy)

            # 오른발 스윙 → col 0~5 = 0
            for j in range(6):
                A_swing[rf_swing, j, :] = 0.0
                A_swing[rf_swing, j, j] = 1.0
                l_swing[rf_swing, j] = 0.0
                u_swing[rf_swing, j] = 0.0

            # 왼발 스윙 → col 6~11 = 0
            for j in range(6):
                A_swing[lf_swing, j, :] = 0.0
                A_swing[lf_swing, j, 6 + j] = 1.0
                l_swing[lf_swing, j] = 0.0
                u_swing[lf_swing, j] = 0.0

            A_ineq = torch.cat([A_ineq, A_swing], dim=1)
            l_ineq = torch.cat([l_ineq, l_swing], dim=1)
            u_ineq = torch.cat([u_ineq, u_swing], dim=1)

            # ForceOptimizer
            F_hat = self.force_opt.solve(K, u_vec, A_ineq, l_ineq, u_ineq)

            # TorqueGen
            tau_ff = self.torque_gen.compute(
                mass_matrix=mass_matrix,
                nle=nle,
                jacobians=jacobians,
                rf_body_idx=rf_body_idx,
                lf_body_idx=lf_body_idx,
                F_hat=F_hat,
            )

            # ── DEBUG: tau_ff / tau_fb 분리 로깅 ──
            self._dbg_cnt = getattr(self, '_dbg_cnt', 0) + 1
            if self._dbg_cnt % 100 == 1:
                print(
                    f"  [WBC] |tau_fb|={tau_fb[0].norm():.2f} max={tau_fb[0].abs().max():.2f}"
                    f"  |tau_ff|={tau_ff[0].norm():.2f} max={tau_ff[0].abs().max():.2f}"
                    f"  |F_hat|={F_hat[0].norm():.2f}"
                )

            return tau_ff + tau_fb
        else:
            # feedforward 인자 없으면 fb만
            return tau_fb

    def compute_dbfc(
        self,
        jacobians: torch.Tensor,     # (B, n_bodies, 6, nv)
        mass_matrix: torch.Tensor,   # (B, nv, nv)
        nle: torch.Tensor,           # (B, nv) — coriolis + gravity
        com_pos: torch.Tensor,       # (B, 3)
        com_vel: torch.Tensor,       # (B, 3)
        com_des: torch.Tensor,       # (B, 3)
        com_dot_des: torch.Tensor,   # (B, 3)
        rf_pos: torch.Tensor,        # (B, 3)
        lf_pos: torch.Tensor,        # (B, 3)
        rf_body_idx: int,
        lf_body_idx: int,
        contact_state: torch.Tensor = None,  # (B, 2) — [right, left]
    ) -> torch.Tensor:
        """DBFC 모드 — C++ computeDBFC() 대응.

        Returns:
            tau: (B, na) 관절 토크
        """
        B = com_pos.shape[0]
        device = self.device
        dtype = self.dtype

        # ── 1. Balance Task: 목표 CoM 가속도 (x,y만 PD, z는 중력만) ──
        ddc_des = self.balance_task.update(com_pos, com_vel, com_des, com_dot_des)
        ddc_des[:, 2] = 0.0

        # ── 2. CoM Dynamics: K*F = u 행렬 구성 ──
        dL = torch.zeros(B, 3, device=device, dtype=dtype)  # 각운동량 변화율 0 가정
        K, u_vec = self.com_dynamics.update(com_pos, lf_pos, rf_pos, ddc_des, dL)

        # ── 3. 제약조건 조립 ──
        # 마찰콘: (B, 5, 3) × 2발 → (B, 10, 12)
        A_fric_r, l_fric_r, u_fric_r = self.friction_cone.update()  # (B, 5, 3)
        A_fric_l, l_fric_l, u_fric_l = self.friction_cone.update()

        # 마찰콘을 12차원으로 확장
        A_fric = torch.zeros(B, 10, 12, device=device, dtype=dtype)
        A_fric[:, :5, :3] = A_fric_r     # 오른발 force
        A_fric[:, 5:, 6:9] = A_fric_l    # 왼발 force
        l_fric = torch.cat([l_fric_r, l_fric_l], dim=-1)  # (B, 10)
        u_fric = torch.cat([u_fric_r, u_fric_l], dim=-1)  # (B, 10)

        # CoP: (B, 8, 12)
        if contact_state is not None:
            A_cop, l_cop, u_cop = self.cop_limits.update(contact_state)
        else:
            A_cop, l_cop, u_cop = self.cop_limits.update(
                torch.ones(B, 2, device=device, dtype=dtype)
            )

        # 결합: (B, 18, 12)
        A_ineq = torch.cat([A_fric, A_cop], dim=1)
        l_ineq = torch.cat([l_fric, l_cop], dim=1)
        u_ineq = torch.cat([u_fric, u_cop], dim=1)

        # 스윙 발 제약 (C++ 대응: 24행 = 10 + 8 + 6)
        A_swing = torch.zeros(B, 6, 12, device=device, dtype=dtype)
        l_swing = torch.full((B, 6), -1e6, device=device, dtype=dtype)
        u_swing = torch.full((B, 6), 1e6, device=device, dtype=dtype)

        if contact_state is not None:
            rf_swing = (contact_state[:, 0] < 0.5)
            lf_swing = (contact_state[:, 1] < 0.5)

            for j in range(6):
                A_swing[:, j, j] = 1.0  # dummy

            for j in range(6):
                A_swing[rf_swing, j, :] = 0.0
                A_swing[rf_swing, j, j] = 1.0
                l_swing[rf_swing, j] = 0.0
                u_swing[rf_swing, j] = 0.0

            for j in range(6):
                A_swing[lf_swing, j, :] = 0.0
                A_swing[lf_swing, j, 6 + j] = 1.0
                l_swing[lf_swing, j] = 0.0
                u_swing[lf_swing, j] = 0.0
        else:
            for j in range(6):
                A_swing[:, j, j] = 1.0

        A_ineq = torch.cat([A_ineq, A_swing], dim=1)
        l_ineq = torch.cat([l_ineq, l_swing], dim=1)
        u_ineq = torch.cat([u_ineq, u_swing], dim=1)

        # ── 4. Force Optimizer: 최적 반력 F_hat ──
        F_hat = self.force_opt.solve(K, u_vec, A_ineq, l_ineq, u_ineq)

        # ── 5. Whole Body Torque: F_hat → tau ──
        tau = self.torque_gen.compute(
            mass_matrix=mass_matrix,
            nle=nle,
            jacobians=jacobians,
            rf_body_idx=rf_body_idx,
            lf_body_idx=lf_body_idx,
            F_hat=F_hat,
        )
        return tau
