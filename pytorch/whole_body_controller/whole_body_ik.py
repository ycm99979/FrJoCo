"""
WholeBodyIK — C++ whole_body_ik.cpp 대응.

Differential IK (Resolved Motion Rate Control), 배치 PyTorch 구현.

스택된 태스크:
  [J_com ]     [dx_com_err ]
  [J_rf  ] dq = [dx_rf_err  ]
  [J_lf  ]     [dx_lf_err  ]

dq = J_stack^T (J_stack J_stack^T + λ²I)^{-1} dx_err / dt
q_des = q_curr + dq * dt

Isaac Lab API:
  - jacobians: robot.root_physx_view.get_jacobians()  → (B, n_bodies, 6, nv)
  - body_pos:  robot.data.body_pos_w                   → (B, n_bodies, 3)
  - body_quat: robot.data.body_quat_w                  → (B, n_bodies, 4)
"""

import torch
from pytorch.config.g1_config import WalkingConfig


def _orientation_error(R_des: torch.Tensor, R_curr: torch.Tensor) -> torch.Tensor:
    """회전 오차 log(R_des @ R_curr^T) → (B, 3).

    Args:
        R_des:  (B, 3, 3)
        R_curr: (B, 3, 3)
    Returns:
        err: (B, 3) axis-angle 벡터
    """
    R_err = R_des @ R_curr.transpose(-1, -2)  # (B, 3, 3)
    # trace
    cos_angle = (R_err[:, 0, 0] + R_err[:, 1, 1] + R_err[:, 2, 2] - 1.0) * 0.5
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)  # (B,)

    # axis from skew-symmetric part
    axis = torch.stack([
        R_err[:, 2, 1] - R_err[:, 1, 2],
        R_err[:, 0, 2] - R_err[:, 2, 0],
        R_err[:, 1, 0] - R_err[:, 0, 1],
    ], dim=-1)  # (B, 3)

    # angle / (2 sin(angle)), 특이점 처리
    denom = 2.0 * torch.sin(angle).unsqueeze(-1).clamp(min=1e-8)
    err = axis * angle.unsqueeze(-1) / denom
    # angle ≈ 0 → err = 0
    small = (angle < 1e-8).unsqueeze(-1)
    err = torch.where(small, torch.zeros_like(err), err)
    return err


class WholeBodyIK:
    """배치 Differential IK — C++ WholeBodyIK 대응.

    Isaac Lab jacobians 텐서를 직접 사용.
    Pinocchio 호출 없이 순수 텐서 연산.
    """

    def __init__(self, cfg: WalkingConfig, nv: int = 29, na: int = 23):
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.nv = nv
        self.na = na
        self.dt = cfg.dt
        self.damping = cfg.ik_damping
        self.v_max = cfg.ik_v_max
        B = cfg.batch_size

        # PD 게인 (actuated joints)
        self.Kp = torch.full((na,), cfg.ik_kp, device=cfg.device, dtype=cfg.dtype)
        self.Kd = torch.full((na,), cfg.ik_kd, device=cfg.device, dtype=cfg.dtype)

        # 결과 버퍼
        self.q_des = None   # (B, nq) — 첫 호출 시 초기화
        self.v_des = torch.zeros(B, nv, device=cfg.device, dtype=cfg.dtype)
        self._first_call = True

    def compute(
        self,
        jacobians: torch.Tensor,     # (B, n_bodies, 6, nv)
        q_curr: torch.Tensor,        # (B, nq)  nq = nv+1 (quaternion)
        dq_curr: torch.Tensor,       # (B, nv)
        com_jac: torch.Tensor,       # (B, 3, nv) — CoM 자코비안
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
        rf_vel_ff: torch.Tensor = None,  # (B, 3)
        lf_vel_ff: torch.Tensor = None,  # (B, 3)
        rf_ori_des: torch.Tensor = None, # (B, 3, 3)
        lf_ori_des: torch.Tensor = None, # (B, 3, 3)
    ):
        """Differential IK 계산 — C++ WholeBodyIK::compute() 대응."""
        B = q_curr.shape[0]
        nv = self.nv
        device = self.device
        dtype = self.dtype
        dt = self.dt

        if self._first_call:
            self.q_des = q_curr.clone()
            self.v_des = torch.zeros(B, nv, device=device, dtype=dtype)
            self._first_call = False

        if rf_vel_ff is None:
            rf_vel_ff = torch.zeros(B, 3, device=device, dtype=dtype)
        if lf_vel_ff is None:
            lf_vel_ff = torch.zeros(B, 3, device=device, dtype=dtype)
        if rf_ori_des is None:
            rf_ori_des = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3)
        if lf_ori_des is None:
            lf_ori_des = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3)

        # ── 1. 태스크 오차 (15,) = CoM(3) + RF(6) + LF(6) ──
        task_dim = 15
        dx_err = torch.zeros(B, task_dim, device=device, dtype=dtype)

        # CoM 위치 오차
        dx_err[:, 0:3] = com_des - com_pos

        # 오른발 위치 + 자세 오차
        dx_err[:, 3:6] = rf_pos_des - rf_pos_curr
        dx_err[:, 6:9] = _orientation_error(rf_ori_des, rf_ori_curr)

        # 왼발 위치 + 자세 오차
        dx_err[:, 9:12] = lf_pos_des - lf_pos_curr
        dx_err[:, 12:15] = _orientation_error(lf_ori_des, lf_ori_curr)

        # ── 2. 자코비안 스택 (B, 15, nv) ──
        J_stack = torch.zeros(B, task_dim, nv, device=device, dtype=dtype)

        # CoM 자코비안 (B, 3, nv)
        J_stack[:, 0:3, :] = com_jac

        # 오른발 자코비안 (B, 6, nv) — Isaac Lab: (B, n_bodies, 6, nv)
        # Isaac Lab 순서: [angular(3), linear(3)] → 우리는 [linear, angular]
        J_rf_raw = jacobians[:, rf_body_idx, :, :]  # (B, 6, nv)
        J_stack[:, 3:6, :] = J_rf_raw[:, 3:6, :]    # linear
        J_stack[:, 6:9, :] = J_rf_raw[:, 0:3, :]    # angular

        # 왼발 자코비안
        J_lf_raw = jacobians[:, lf_body_idx, :, :]  # (B, 6, nv)
        J_stack[:, 9:12, :] = J_lf_raw[:, 3:6, :]   # linear
        J_stack[:, 12:15, :] = J_lf_raw[:, 0:3, :]  # angular

        # ── 3. Damped Pseudo-Inverse ──
        # dx_err 클램핑: 너무 큰 오차는 제한 (시뮬레이터 안정성)
        dx_err_clamped = dx_err.clamp(-0.05, 0.05)  # 최대 5cm/rad 오차
        dx_rate = dx_err_clamped / dt  # (B, 15)

        # feedforward velocity 추가
        dx_rate[:, 3:6] += rf_vel_ff
        dx_rate[:, 9:12] += lf_vel_ff

        # JJt = J @ J^T + λ²I  → (B, 15, 15)
        JJt = torch.bmm(J_stack, J_stack.transpose(-1, -2))
        JJt += self.damping * torch.eye(task_dim, device=device, dtype=dtype).unsqueeze(0)

        # v_des = J^T @ (JJt)^{-1} @ dx_rate
        # (B, 15, 1)
        dx_rate_col = dx_rate.unsqueeze(-1)
        solved = torch.linalg.solve(JJt, dx_rate_col)  # (B, 15, 1)
        self.v_des = torch.bmm(J_stack.transpose(-1, -2), solved).squeeze(-1)  # (B, nv)

        # v_des 클램핑 (actuated joints만, floating base는 q_des 적분에만 사용)
        self.v_des[:, 6:] = self.v_des[:, 6:].clamp(-self.v_max, self.v_max)

        # ── 4. 관절 적분 q_des = q_curr + v_des * dt ──
        # q_des는 매 스텝 q_curr 기준으로 적분 (누적 오차 방지)
        self.q_des = q_curr.clone()

        # floating base 위치 (0:3)
        self.q_des[:, :3] += self.v_des[:, :3] * dt

        # floating base 자세 (quaternion q[3:7], v[3:6])
        omega_dt = self.v_des[:, 3:6] * dt  # (B, 3)
        angle = omega_dt.norm(dim=-1, keepdim=True)  # (B, 1)
        axis = omega_dt / angle.clamp(min=1e-10)  # (B, 3)
        half = angle * 0.5
        dq_w = torch.cos(half)       # (B, 1)
        dq_xyz = axis * torch.sin(half)  # (B, 3)

        # 현재 quaternion: Isaac Lab 순서 [x, y, z, w] → q_curr[:, 3:7]
        qx, qy, qz, qw = q_curr[:, 3], q_curr[:, 4], q_curr[:, 5], q_curr[:, 6]
        # dq quaternion
        dx, dy, dz, dw = dq_xyz[:, 0], dq_xyz[:, 1], dq_xyz[:, 2], dq_w.squeeze(-1)

        # quaternion 곱: dq * q_fb
        new_w = dw * qw - dx * qx - dy * qy - dz * qz
        new_x = dw * qx + dx * qw + dy * qz - dz * qy
        new_y = dw * qy - dx * qz + dy * qw + dz * qx
        new_z = dw * qz + dx * qy - dy * qx + dz * qw

        norm = torch.sqrt(new_x**2 + new_y**2 + new_z**2 + new_w**2).clamp(min=1e-10)
        self.q_des[:, 3] = new_x / norm
        self.q_des[:, 4] = new_y / norm
        self.q_des[:, 5] = new_z / norm
        self.q_des[:, 6] = new_w / norm

        # actuated joints (7:nq)
        self.q_des[:, 7:] += self.v_des[:, 6:] * dt

    def compute_pd_torque(
        self,
        q_curr: torch.Tensor,   # (B, nq)
        v_curr: torch.Tensor,   # (B, nv)
    ) -> torch.Tensor:
        """PD 토크 계산 (actuated joints만) — C++ computePDTorque() 대응.

        Returns:
            tau: (B, na)
        """
        q_act_err = self.q_des[:, 7:] - q_curr[:, 7:]
        # v_des[:, :6]은 floating base → PD에 사용 안 함, actuated(6:)만 사용
        v_act_err = self.v_des[:, 6:] - v_curr[:, 6:]

        # DEBUG: 첫 몇 스텝에서 오차 확인
        self._pd_dbg_cnt = getattr(self, '_pd_dbg_cnt', 0) + 1
        if self._pd_dbg_cnt <= 3 or self._pd_dbg_cnt % 500 == 0:
            print(f"  [PD dbg step={self._pd_dbg_cnt}]")
            print(f"    q_des[7:12] = {self.q_des[0, 7:12].tolist()}")
            print(f"    q_cur[7:12] = {q_curr[0, 7:12].tolist()}")
            print(f"    q_err[:5]   = {q_act_err[0, :5].tolist()}")
            print(f"    |q_err|     = {q_act_err[0].norm():.4f}")
            print(f"    |v_err|     = {v_act_err[0].norm():.4f}")
            print(f"    quat q_des[3:7] = {self.q_des[0, 3:7].tolist()}")
            print(f"    quat q_cur[3:7] = {q_curr[0, 3:7].tolist()}")

        return self.Kp * q_act_err + self.Kd * v_act_err
