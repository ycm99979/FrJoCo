# Dynamic Balance Force Control for Compliant Humanoid Robots
# II. COM Dynamics model의 식 (9) ~(10) 까지 구현

import torch
from pytorch.config.g1_config import WalkingConfig


class BatchedCoPLimits:
    """Center of Pressure 제약 — 양발 각각 4개씩, 총 8행 × 12열."""

    def __init__(self, cfg: WalkingConfig, dX_max, dX_min, dY_max, dY_min):
        self.num_envs = cfg.batch_size
        self.device = cfg.device

        A = torch.zeros((8, 12), device=cfg.device, dtype=cfg.dtype)

        # 오른발 CoP 제약 (force 인덱스: fx=0, fy=1, fz=2, tx=3, ty=4, tz=5)
        A[0, 3], A[0, 2] = 1.0, -dY_max
        A[1, 3], A[1, 2] = 1.0, -dY_min
        A[2, 4], A[2, 2] = -1.0, -dX_max
        A[3, 4], A[3, 2] = -1.0, -dX_min

        # 왼발 CoP 제약 (force 인덱스: fx=6, fy=7, fz=8, tx=9, ty=10, tz=11)
        A[4, 9],  A[4, 8] = 1.0, -dY_max
        A[5, 9],  A[5, 8] = 1.0, -dY_min
        A[6, 10], A[6, 8] = -1.0, -dX_max
        A[7, 10], A[7, 8] = -1.0, -dX_min

        self.A = A.unsqueeze(0).expand(cfg.batch_size, -1, -1)

        INF = float('inf')
        self.l = torch.tensor(
            [-INF, 0.0, -INF, 0.0, -INF, 0.0, -INF, 0.0], device=cfg.device, dtype=cfg.dtype
        ).expand(cfg.batch_size, -1)
        self.u = torch.tensor(
            [0.0, INF, 0.0, INF, 0.0, INF, 0.0, INF], device=cfg.device, dtype=cfg.dtype
        ).expand(cfg.batch_size, -1)

    def update(self, contact_state):
        """contact_state: (num_envs, 2) — [right, left] 접촉 여부 (1=stance, 0=swing)."""
        l = self.l.clone()
        u = self.u.clone()

        # 오른발 swing → 해당 행 경계를 0으로
        rf_swing = (contact_state[:, 0] < 0.5).unsqueeze(1)
        l[:, 0:4] = torch.where(rf_swing, torch.zeros_like(l[:, 0:4]), l[:, 0:4])
        u[:, 0:4] = torch.where(rf_swing, torch.zeros_like(u[:, 0:4]), u[:, 0:4])

        # 왼발 swing
        lf_swing = (contact_state[:, 1] < 0.5).unsqueeze(1)
        l[:, 4:8] = torch.where(lf_swing, torch.zeros_like(l[:, 4:8]), l[:, 4:8])
        u[:, 4:8] = torch.where(lf_swing, torch.zeros_like(u[:, 4:8]), u[:, 4:8])

        return self.A, l, u
