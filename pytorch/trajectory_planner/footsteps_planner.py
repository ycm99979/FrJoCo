"""
배치 Footstep Planner — PyTorch 벡터화

기존 mujoco Layer1.plan_footsteps()를 배치 텐서로 변환.
for 루프 없이 torch.arange + broadcasting으로 전체 footstep 시퀀스를 한번에 생성.

입력:  init_com_xy (B, 2)
출력:  footsteps   (B, N, 2)  — ZMP/DCM 레퍼런스용 발 착지 위치
"""

import torch
from typing import Optional
from pytorch.config.g1_config import WalkingConfig


class FootstepPlanner:
    """배치 footstep 시퀀스 생성기.

    기존 코드 매핑:
      - DCM Layer1: footsteps[0] = (init_x, +width), 이후 좌우 교대
      - ZMP planner: footsteps[0] = 양발 중심(DSP), 이후 좌우 교대

    여기서는 DCM 방식을 기본으로 하되, ZMP 방식도 지원.
    """

    def __init__(self, cfg: WalkingConfig):
        self.cfg = cfg

    def plan(
        self,
        init_com_xy: torch.Tensor,          # (B, 2)
        n_steps: Optional[int] = None,
        step_length: Optional[float] = None,
        step_width: Optional[float] = None,
    ) -> torch.Tensor:
        """배치 footstep 생성.

        Args:
            init_com_xy: (B, 2) 초기 CoM XY 위치
            n_steps: 걸음 수 (None이면 cfg 사용)
            step_length: 보폭 (None이면 cfg 사용)
            step_width: 좌우 간격 (None이면 cfg 사용)

        Returns:
            footsteps: (B, N, 2) 각 스텝의 ZMP 레퍼런스 위치
        """
        cfg = self.cfg
        N = n_steps or cfg.n_steps
        sl = step_length or cfg.step_length
        sw = step_width or cfg.step_width
        B = init_com_xy.shape[0]
        device = init_com_xy.device
        dtype = init_com_xy.dtype

        # 스텝 인덱스: (N,)
        idx = torch.arange(N, device=device, dtype=dtype)

        # ── X 좌표: step 0은 init_x, 이후 init_x + i * step_length ──
        # x_offsets: (N,) — [0, sl, 2*sl, ...]
        # 첫 스텝은 제자리이므로 offset 0
        x_offsets = torch.where(idx == 0, torch.zeros_like(idx), idx * sl)
        # (B, N) = (B, 1) + (1, N)
        x = init_com_xy[:, 0:1] + x_offsets.unsqueeze(0)

        # ── Y 좌표: 홀수 스텝 = -width, 짝수 스텝 = +width ──
        # 첫 스텝(i=0)은 왼발(+width)
        # sign: (N,) — [+1, -1, +1, -1, ...]
        sign = torch.where(idx % 2 == 0,
                           torch.ones_like(idx),
                           -torch.ones_like(idx))
        y = sign.unsqueeze(0) * sw  # (1, N) broadcast → (B, N)
        y = y.expand(B, N)

        # (B, N, 2)
        footsteps = torch.stack([x, y], dim=-1)
        return footsteps
