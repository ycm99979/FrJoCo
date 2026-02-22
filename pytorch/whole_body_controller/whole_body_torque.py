"""
WholeBodyTorqueGenerator — C++ whole_body_torque.cpp 대응.

논문 식 (19-20): 최적 지면 반력 F_hat → 관절 토크.

전신 동역학:
  M(q)*ddq + h(q,dq) = S^T * tau + J_c^T * F

결정변수: z = [ddq (nv), tau (na)]
등식 제약: G * z = f
  G = [M   -S^T]   f = [J_c^T * F - h]
      [J_c   0 ]       [-Jdot*qdot   ]

최소화: z = (G^T G + W)^{-1} G^T f

Isaac Lab API:
  - mass_matrix: robot.root_physx_view.get_generalized_mass_matrices() → (B, nv, nv)
  - gravity_comp: robot.root_physx_view.get_gravity_compensation_forces() → (B, nv)
  - jacobians: robot.root_physx_view.get_jacobians() → (B, n_bodies, 6, nv)

배치 PyTorch 구현.
"""

import torch
from pytorch.config.g1_config import WalkingConfig


class WholeBodyTorqueGenerator:
    """전신 토크 생성기 — C++ WholeBodyTorqueGenerator 대응.

    nv = 29 (floating base 6 + joints 23)
    na = 23 (actuated joints)
    nc = 12 (양발 × 6DoF 접촉)
    """

    def __init__(self, cfg: WalkingConfig, nv: int = 29, na: int = 23, nc: int = 12):
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.nv = nv
        self.na = na
        self.nc = nc

        # 가중치 행렬 W: (nv+na, nv+na)
        W = torch.zeros(nv + na, nv + na, device=cfg.device, dtype=cfg.dtype)
        W[:nv, :nv] = cfg.wbt_w_ddq * torch.eye(nv, device=cfg.device, dtype=cfg.dtype)
        W[nv:, nv:] = cfg.wbt_w_tau * torch.eye(na, device=cfg.device, dtype=cfg.dtype)
        self.W = W

        # -S^T 블록 (고정): G[6:nv, nv:nv+na] = -I
        # floating base(0:6)는 actuator 없음, actuated(6:nv)에 -I
        self._S_T_neg = torch.zeros(nv, na, device=cfg.device, dtype=cfg.dtype)
        self._S_T_neg[6:, :] = -torch.eye(na, device=cfg.device, dtype=cfg.dtype)

    def compute(
        self,
        mass_matrix: torch.Tensor,    # (B, nv, nv)
        nle: torch.Tensor,            # (B, nv) — nonlinear effects (coriolis + gravity)
        jacobians: torch.Tensor,      # (B, n_bodies, 6, nv)
        rf_body_idx: int,
        lf_body_idx: int,
        F_hat: torch.Tensor,          # (B, 12) — 최적 지면 반력
    ) -> torch.Tensor:
        """F_hat → 관절 토크 tau.

        Returns:
            tau: (B, na)
        """
        B = mass_matrix.shape[0]
        nv, na, nc = self.nv, self.na, self.nc
        device = self.device
        dtype = self.dtype

        # ── 1. 접촉 자코비안 J_c (B, nc, nv) ──
        # C++ Pinocchio LOCAL_WORLD_ALIGNED: [linear(3), angular(3)]
        # MuJoCo jacobians: [angular(3), linear(3)]
        # → 순서 변환 필요
        J_c = torch.zeros(B, nc, nv, device=device, dtype=dtype)
        # 오른발 (6 × nv) — [linear, angular] 순서로 재배치
        J_rf = jacobians[:, rf_body_idx, :, :]  # (B, 6, nv) — [ang(0:3), lin(3:6)]
        J_c[:, 0:3, :] = J_rf[:, 3:6, :]  # linear
        J_c[:, 3:6, :] = J_rf[:, 0:3, :]  # angular
        # 왼발 (6 × nv)
        J_lf = jacobians[:, lf_body_idx, :, :]
        J_c[:, 6:9, :]  = J_lf[:, 3:6, :]  # linear
        J_c[:, 9:12, :] = J_lf[:, 0:3, :]  # angular

        # ── 2. G 행렬 조립: (nv+nc) × (nv+na) ──
        G = torch.zeros(B, nv + nc, nv + na, device=device, dtype=dtype)

        # 상단: [M, -S^T]
        G[:, :nv, :nv] = mass_matrix
        G[:, :nv, nv:] = self._S_T_neg.unsqueeze(0).expand(B, -1, -1)

        # 하단: [J_c, 0]
        G[:, nv:, :nv] = J_c

        # ── 3. f 벡터: (nv+nc,) ──
        f = torch.zeros(B, nv + nc, device=device, dtype=dtype)

        # f[:nv] = -h + J_c^T @ F_hat
        # F_hat: (B, 12) → (B, 12, 1)
        Jc_T_F = torch.bmm(J_c.transpose(-1, -2), F_hat.unsqueeze(-1)).squeeze(-1)  # (B, nv)
        f[:, :nv] = -nle + Jc_T_F

        # f[nv:] = -Jdot*qdot ≈ 0 (접촉 발 가속도 0 가정)
        # Isaac Lab에서 Jdot*qdot를 직접 제공하지 않으므로 0으로 근사
        # (정밀 구현 시 finite difference 또는 별도 계산 필요)

        # ── 4. z = (G^T G + W)^{-1} G^T f ──
        # (B, nv+na, nv+nc) @ (B, nv+nc, 1) → (B, nv+na, 1)
        Gt = G.transpose(-1, -2)  # (B, nv+na, nv+nc)
        GtG = torch.bmm(Gt, G)   # (B, nv+na, nv+na)
        GtG_W = GtG + self.W.unsqueeze(0)  # (B, nv+na, nv+na)
        Gt_f = torch.bmm(Gt, f.unsqueeze(-1))  # (B, nv+na, 1)

        z = torch.linalg.solve(GtG_W, Gt_f).squeeze(-1)  # (B, nv+na)

        # z = [ddq(nv), tau(na)] → tau
        tau = z[:, nv:]  # (B, na)
        return tau
