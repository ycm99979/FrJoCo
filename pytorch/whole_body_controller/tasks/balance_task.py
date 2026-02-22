"""
BalanceTask — C++ balance_task.cpp 대응.

논문 식 (24): CoM PD 제어 → 목표 가속도 생성
  ddc_des = Kp * (c_des - c) + Kd * (dc_des - dc)

배치 텐서 연산 (B, 3).
"""

import torch
from pytorch.config.g1_config import WalkingConfig


class BalanceTask:
    """CoM PD 밸런스 태스크 — C++ BalanceTask 대응."""

    def __init__(self, cfg: WalkingConfig):
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.kp = cfg.bal_kp
        self.kd = cfg.bal_kd

    def update(
        self,
        com_curr: torch.Tensor,      # (B, 3)
        com_dot_curr: torch.Tensor,   # (B, 3)
        com_des: torch.Tensor,        # (B, 3)
        com_dot_des: torch.Tensor = None,  # (B, 3)
    ) -> torch.Tensor:
        """목표 CoM 가속도 계산.

        Returns:
            ddc_des: (B, 3)
        """
        if com_dot_des is None:
            com_dot_des = torch.zeros_like(com_curr)
        return self.kp * (com_des - com_curr) + self.kd * (com_dot_des - com_dot_curr)
