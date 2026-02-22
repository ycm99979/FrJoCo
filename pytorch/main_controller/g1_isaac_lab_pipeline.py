"""
G1 Walking Controller Pipeline — C++ g1_walking_controller 대응.

구조:
  생성자: footstep plan → ZMP ref → foot trajectory → CoM ref (오프라인)
  mpc_loop(100Hz): 상태 보정 → MPC QP → 상태 업데이트
  wbc_loop(1kHz):  IK → ForceOptimizer → TorqueGenerator → τ
  standing_loop:   MPC 없이 초기 자세 유지

Isaac Lab API 사용:
  - robot.root_physx_view.get_jacobians()
  - robot.root_physx_view.get_generalized_mass_matrices()
  - robot.root_physx_view.get_gravity_compensation_forces()
"""

import torch
from pytorch.config.g1_config import WalkingConfig
from pytorch.dynamics_model.LIPM_dynamics import LIPMdynamics
from pytorch.controller.LIPM_MPC import LIPM_MPC
from pytorch.trajectory_planner.zmp_trajectory import ZmpTrajectory
from pytorch.trajectory_planner.foot_trajectory import FootTrajectoryGenerator
from pytorch.trajectory_planner.footsteps_planner import FootstepPlanner
from pytorch.utils.batch_utils import get_contact_state
from pytorch.whole_body_controller.DBFC import WBC


class G1WalkingPipeline:
    """G1 보행 컨트롤러 파이프라인.

    C++ G1WalkingController 대응.
    오프라인 궤적 생성 + 온라인 MPC/WBC 루프.
    """

    def __init__(
        self,
        cfg: WalkingConfig,
        init_com: torch.Tensor,     # (B, 3)
        init_lf: torch.Tensor,      # (B, 3)
        init_rf: torch.Tensor,      # (B, 3)
        nv: int = 43,               # 일반화 속도 차원 (6 + n_joints)
        na: int = 37,               # actuated joints 수
    ):
        self.cfg = cfg
        B = cfg.batch_size
        device = cfg.device
        dtype = cfg.dtype

        # 초기 목표 (standing 모드용) — 반드시 cfg.device로 이동
        self.init_com = init_com.to(device=device, dtype=dtype).clone()
        self.init_lf = init_lf.to(device=device, dtype=dtype).clone()
        self.init_rf = init_rf.to(device=device, dtype=dtype).clone()

        # ── 모듈 초기화 ──
        self.lipm = LIPMdynamics(cfg)
        self.mpc = LIPM_MPC(cfg)
        self.zmp_traj = ZmpTrajectory(cfg)
        self.foot_traj_gen = FootTrajectoryGenerator(cfg)
        self.footstep_planner = FootstepPlanner(cfg)
        self.wbc = WBC(cfg, nv=nv, na=na)

        # ── MPC 상태 [pos, vel, acc] per axis ──
        self.x_state = torch.zeros(B, 3, device=device, dtype=dtype)
        self.y_state = torch.zeros(B, 3, device=device, dtype=dtype)
        self.x_state[:, 0] = init_com[:, 0]
        self.y_state[:, 0] = init_com[:, 1]

        # ── 오프라인 궤적 생성 ──
        self._plan_trajectories()

        # 인덱스
        self.traj_idx = 0

    def _plan_trajectories(self):
        """오프라인 궤적 생성 — C++ 생성자 대응."""
        cfg = self.cfg
        B = cfg.batch_size
        device = cfg.device
        dtype = cfg.dtype

        # 1. 발자국 계획
        init_com_xy = self.init_com[:, :2]  # (B, 2)
        self.footsteps = self.footstep_planner.plan(init_com_xy)  # (B, N, 2)

        # 2. ZMP ref 배열 (MPC_DT 해상도)
        self.zmp_traj.build_zmp_reference(self.footsteps)

        # 3. 발 궤적 (WBC_DT 해상도, pos+vel+acc)
        self.foot_result = self.foot_traj_gen.generate(
            self.footsteps, self.init_lf, self.init_rf
        )
        # foot_result: dict with left_pos, right_pos, left_vel, right_vel, left_acc, right_acc
        # 각각 (B, T, 3)

        # 4. CoM ref 궤적 (오프라인 MPC forward simulation)
        walk_samples = self.zmp_traj.walk_samples
        self.com_ref = torch.zeros(B, walk_samples, 2, device=device, dtype=dtype)

        # LIPM base 행렬
        A = self.lipm.Ad[0]  # (3, 3)
        B_mat = self.lipm.Bd[0]  # (3, 1)

        x_sim = torch.zeros(B, 3, device=device, dtype=dtype)
        y_sim = torch.zeros(B, 3, device=device, dtype=dtype)
        x_sim[:, 0] = self.init_com[:, 0]
        y_sim[:, 0] = self.init_com[:, 1]

        for i in range(walk_samples):
            self.com_ref[:, i, 0] = x_sim[:, 0]
            self.com_ref[:, i, 1] = y_sim[:, 0]

            # ZMP ref slice
            zmp_slice = self.zmp_traj.get_zmp_ref_slice(i, cfg.mpc_horizon)  # (B, N, 2)
            zmp_ref_x = zmp_slice[:, :, 0]  # (B, N)
            zmp_ref_y = zmp_slice[:, :, 1]  # (B, N)

            # MPC solve
            jerk_x, jerk_y = self.mpc.solve(x_sim, y_sim, zmp_ref_x, zmp_ref_y)

            # 상태 업데이트: x = A @ x + B * u
            # x_sim: (B, 3) → (B, 3, 1) for bmm
            x_sim = (A @ x_sim.unsqueeze(-1)).squeeze(-1) + B_mat.squeeze(-1) * jerk_x.unsqueeze(-1)
            y_sim = (A @ y_sim.unsqueeze(-1)).squeeze(-1) + B_mat.squeeze(-1) * jerk_y.unsqueeze(-1)

    # ── 시간 → 인덱스 변환 ──

    def get_mpc_index(self, t: float) -> int:
        """MPC_DT 해상도 인덱스."""
        return min(int(t / self.cfg.mpc_dt), self.zmp_traj.walk_samples - 1)

    def get_wbc_index(self, t: float) -> int:
        """WBC_DT 해상도 인덱스."""
        total = self.foot_result['left_pos'].shape[1]
        return min(int(t / self.cfg.dt), total - 1)

    # ── MPC 100Hz ──

    def mpc_loop(
        self,
        com_pos: torch.Tensor,   # (B, 3)
        com_vel: torch.Tensor,   # (B, 3)
        sim_time: float,
    ):
        """MPC 루프 — C++ mpcLoop() 대응.

        실측값으로 상태 보정 → MPC QP → 상태 업데이트.
        """
        idx = self.get_mpc_index(sim_time)
        if idx >= self.zmp_traj.walk_samples:
            return
        self.traj_idx = idx

        # 실측값으로 상태 보정
        self.x_state[:, 0] = com_pos[:, 0]
        self.x_state[:, 1] = com_vel[:, 0]
        self.y_state[:, 0] = com_pos[:, 1]
        self.y_state[:, 1] = com_vel[:, 1]

        # 가속도 보정: ddx = omega^2 * (x - zmp_ref)
        omega2 = self.cfg.gravity / self.cfg.com_height
        zmp_ref = self.zmp_traj.zmp_ref  # (B, M, 2)
        self.x_state[:, 2] = omega2 * (self.x_state[:, 0] - zmp_ref[:, idx, 0])
        self.y_state[:, 2] = omega2 * (self.y_state[:, 0] - zmp_ref[:, idx, 1])

        # MPC QP
        zmp_slice = self.zmp_traj.get_zmp_ref_slice(idx, self.cfg.mpc_horizon)
        jerk_x, jerk_y = self.mpc.solve(
            self.x_state, self.y_state,
            zmp_slice[:, :, 0], zmp_slice[:, :, 1]
        )

        # 상태 업데이트
        A = self.lipm.Ad[0]
        B_mat = self.lipm.Bd[0].squeeze(-1)
        self.x_state = (A @ self.x_state.unsqueeze(-1)).squeeze(-1) + B_mat * jerk_x.unsqueeze(-1)
        self.y_state = (A @ self.y_state.unsqueeze(-1)).squeeze(-1) + B_mat * jerk_y.unsqueeze(-1)

    # ── WBC 1kHz ──

    def wbc_loop(
        self,
        jacobians: torch.Tensor,     # (B, n_bodies, 6, nv)
        mass_matrix: torch.Tensor,   # (B, nv, nv)
        gravity_comp: torch.Tensor,  # (B, nv)
        joint_pos: torch.Tensor,     # (B, n_joints)
        joint_vel: torch.Tensor,     # (B, n_joints)
        com_pos: torch.Tensor,       # (B, 3)
        com_vel: torch.Tensor,       # (B, 3)
        lf_body_idx: int,
        rf_body_idx: int,
        sim_time: float,
        com_jac: torch.Tensor = None,  # (B, 3, nv) — CoM 자코비안
        q_full: torch.Tensor = None,   # (B, nq) — 전체 일반화 좌표 (floating base 포함)
        dq_full: torch.Tensor = None,  # (B, nv) — 전체 일반화 속도
        rf_pos_curr: torch.Tensor = None,  # (B, 3)
        lf_pos_curr: torch.Tensor = None,  # (B, 3)
        rf_ori_curr: torch.Tensor = None,  # (B, 3, 3)
        lf_ori_curr: torch.Tensor = None,  # (B, 3, 3)
        use_dbfc: bool = False,
    ) -> torch.Tensor:
        """WBC 루프 — C++ wbcLoop() 대응.

        use_dbfc=False: IK + PD 모드 (경로 A)
        use_dbfc=True:  DBFC 모드 (경로 B, BalanceTask → ForceOpt → Torque)

        IK 모드에 필요한 인자: jacobians, q_full, dq_full, com_jac, com_pos,
            rf/lf_body_idx, rf/lf_pos_curr, rf/lf_ori_curr
        DBFC 모드에 필요한 인자: jacobians, mass_matrix, gravity_comp, com_pos, com_vel,
            rf/lf_pos_curr, rf/lf_body_idx

        Returns:
            tau: (B, na) 관절 토크
        """
        cfg = self.cfg
        wbc_idx = self.get_wbc_index(sim_time)

        # 목표 CoM
        com_des = torch.stack([
            self.x_state[:, 0],
            self.y_state[:, 0],
            torch.full((cfg.batch_size,), cfg.com_height, device=cfg.device, dtype=cfg.dtype),
        ], dim=-1)  # (B, 3)

        # 목표 발 위치/속도 (WBC_DT 해상도 궤적에서 읽기)
        lf_pos_des = self.foot_result['left_pos'][:, wbc_idx, :]   # (B, 3)
        rf_pos_des = self.foot_result['right_pos'][:, wbc_idx, :]  # (B, 3)
        lf_vel_ff = self.foot_result['left_vel'][:, wbc_idx, :]    # (B, 3)
        rf_vel_ff = self.foot_result['right_vel'][:, wbc_idx, :]   # (B, 3)

        if use_dbfc and rf_pos_curr is not None:
            # ── DBFC 모드 (경로 B) ──
            contact = self.get_contact_state(sim_time)
            com_dot_des = torch.zeros_like(com_des)
            tau = self.wbc.compute_dbfc(
                jacobians=jacobians,
                mass_matrix=mass_matrix,
                nle=gravity_comp,
                com_pos=com_pos,
                com_vel=com_vel,
                com_des=com_des,
                com_dot_des=com_dot_des,
                rf_pos=rf_pos_curr,
                lf_pos=lf_pos_curr,
                rf_body_idx=rf_body_idx,
                lf_body_idx=lf_body_idx,
                contact_state=contact,
            )
        elif q_full is not None and com_jac is not None:
            # ── IK(fb) + ForceOpt+TorqueGen(ff) 모드 (경로 A) ──
            contact = self.get_contact_state(sim_time)
            tau = self.wbc.compute_ik(
                jacobians=jacobians,
                q_curr=q_full,
                dq_curr=dq_full,
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
                rf_ori_des=rf_ori_curr,   # 현재 자세 유지 (C++ 대응)
                lf_ori_des=lf_ori_curr,
                mass_matrix=mass_matrix,
                nle=gravity_comp,
                com_vel=com_vel,
                contact_state=contact,
                mpc_ddx=self.x_state[0, 2].item(),
                mpc_ddy=self.y_state[0, 2].item(),
            )
        else:
            # ── Fallback: gravity compensation ──
            n_joints = joint_pos.shape[1]
            tau = gravity_comp[:, 6:]
            if tau.shape[1] > n_joints:
                tau = tau[:, :n_joints]

        return tau

    # ── Standing Balance ──

    def standing_loop(
        self,
        jacobians: torch.Tensor,
        mass_matrix: torch.Tensor,
        gravity_comp: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        com_pos: torch.Tensor,
        com_vel: torch.Tensor,
        lf_body_idx: int,
        rf_body_idx: int,
        com_jac: torch.Tensor = None,
        q_full: torch.Tensor = None,
        dq_full: torch.Tensor = None,
        rf_pos_curr: torch.Tensor = None,
        lf_pos_curr: torch.Tensor = None,
        rf_ori_curr: torch.Tensor = None,
        lf_ori_curr: torch.Tensor = None,
    ) -> torch.Tensor:
        """Standing balance — C++ standingLoop() 대응.

        MPC 없이 초기 자세 유지. IK + PD.
        IK 인자가 없으면 gravity comp fallback.
        """
        cfg = self.cfg
        com_des = self.init_com.clone()

        if q_full is not None and com_jac is not None:
            tau = self.wbc.compute_ik(
                jacobians=jacobians,
                q_curr=q_full,
                dq_curr=dq_full,
                com_jac=com_jac,
                com_pos=com_pos,
                rf_body_idx=rf_body_idx,
                lf_body_idx=lf_body_idx,
                com_des=com_des,
                rf_pos_des=self.init_rf,
                lf_pos_des=self.init_lf,
                rf_pos_curr=rf_pos_curr,
                lf_pos_curr=lf_pos_curr,
                rf_ori_curr=rf_ori_curr,
                lf_ori_curr=lf_ori_curr,
                rf_ori_des=rf_ori_curr,   # 현재 자세 유지 (C++ 대응)
                lf_ori_des=lf_ori_curr,
                mass_matrix=mass_matrix,
                nle=gravity_comp,
                com_vel=com_vel,
                contact_state=torch.ones(cfg.batch_size, 2, device=cfg.device, dtype=cfg.dtype),
            )
        else:
            n_joints = joint_pos.shape[1]
            tau = gravity_comp[:, 6:]
            if tau.shape[1] > n_joints:
                tau = tau[:, :n_joints]
        return tau

    # ── 접촉 상태 판별 ──

    def get_contact_state(self, sim_time: float) -> torch.Tensor:
        """접촉 상태 판별 — batch_utils.get_contact_state 위임."""
        cfg = self.cfg
        return get_contact_state(
            sim_time=sim_time,
            batch_size=cfg.batch_size,
            n_steps=cfg.n_steps,
            step_time=cfg.step_time,
            dsp_time=cfg.dsp_time,
            device=cfg.device,
            dtype=cfg.dtype,
        )
