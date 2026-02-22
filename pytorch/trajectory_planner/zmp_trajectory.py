"""
배치 ZMP Reference 궤적 — PyTorch 벡터화

C++ FrMoCo zmp_trajectory.cpp 대응:
  - planFootsteps → footsteps_planner.py로 분리 (이미 존재)
  - generateZmpRef → build_zmp_reference()
  - getZmpRefSlice → get_zmp_ref_slice()

LIPM 상태공간은 dynamics_model/LIPM_dynamics.py에,
MPC QP 풀이는 controller/LIPM_MPC.py에 분리.
"""

import torch
from pytorch.config.g1_config import WalkingConfig


class ZmpTrajectory:
    """ZMP Reference 배열 생성기.

    C++ ZmpTrajectory 클래스 대응.
    발자국 시퀀스 → 전체 ZMP ref 배열 생성.
    """

    def __init__(self, cfg: WalkingConfig):
        self.cfg = cfg
        # 생성 후 build_zmp_reference()로 채워짐
        self.zmp_ref: torch.Tensor | None = None  # (B, M, 2)
        self.total_samples = 0
        self.walk_samples = 0

    def build_zmp_reference(
        self,
        footsteps: torch.Tensor,  # (B, N_steps, 2)
    ) -> torch.Tensor:            # (B, M, 2)
        """Footstep 시퀀스 → ZMP 레퍼런스 배열.

        C++ generateZmpRef() 대응:
          - DSP 구간: 이전 발 → 현재 발로 코사인 보간 (ramp)
          - SSP 구간: 현재 발 위치에 고정
          - 첫 스텝: 전체 구간 첫 발 위치 고정
          - 마지막 발 이후: 마지막 발 좌표 유지 + preview 여유분
        """
        cfg = self.cfg
        B, N_steps, _ = footsteps.shape
        # ZMP ref는 MPC 해상도 (mpc_dt)로 생성 — C++ 대응
        mpc_samples_per_step = round(cfg.step_time / cfg.mpc_dt)
        dsp_samples = round(cfg.dsp_time / cfg.mpc_dt)
        horizon = cfg.mpc_horizon
        device = footsteps.device
        dtype = footsteps.dtype

        self.walk_samples = N_steps * mpc_samples_per_step
        self.total_samples = self.walk_samples + horizon
        M = self.total_samples

        zmp_ref = torch.zeros(B, M, 2, device=device, dtype=dtype)

        for i in range(N_steps):
            t_start = i * mpc_samples_per_step
            t_end = min((i + 1) * mpc_samples_per_step, M)

            if i == 0:
                # 첫 스텝: 전체 구간 첫 발 위치 고정
                zmp_ref[:, t_start:t_end, :] = footsteps[:, 0:1, :].expand(B, t_end - t_start, 2)
            else:
                prev = footsteps[:, i - 1, :]  # (B, 2)
                curr = footsteps[:, i, :]      # (B, 2)

                # DSP: 코사인 보간 ramp
                for k in range(dsp_samples):
                    idx = t_start + k
                    if idx >= M:
                        break
                    alpha = 0.5 * (1.0 - torch.cos(torch.tensor(torch.pi * k / dsp_samples, device=device, dtype=dtype)))
                    zmp_ref[:, idx, :] = (1.0 - alpha) * prev + alpha * curr

                # SSP: 현재 발 위치 고정
                ssp_start = t_start + dsp_samples
                if ssp_start < t_end:
                    zmp_ref[:, ssp_start:t_end, :] = curr.unsqueeze(1).expand(B, t_end - ssp_start, 2)

        # 마지막 발 이후 유지
        last_filled = N_steps * mpc_samples_per_step
        if last_filled < M:
            zmp_ref[:, last_filled:, :] = footsteps[:, -1:, :].expand(B, M - last_filled, 2)

        self.zmp_ref = zmp_ref
        return zmp_ref

    def get_zmp_ref_slice(
        self,
        current_idx: int,
        horizon: int,
    ) -> torch.Tensor:  # (B, horizon, 2)
        """현재 인덱스 기준으로 horizon개의 ZMP ref 슬라이스 반환.

        C++ getZmpRefSlice() 대응.
        범위 초과 시 마지막 값으로 패딩.
        """
        assert self.zmp_ref is not None, "build_zmp_reference()를 먼저 호출"
        M = self.zmp_ref.shape[1]
        end_idx = min(current_idx + horizon, M)
        sliced = self.zmp_ref[:, current_idx:end_idx, :]  # (B, ≤horizon, 2)

        actual_len = sliced.shape[1]
        if actual_len < horizon:
            pad = self.zmp_ref[:, -1:, :].expand(-1, horizon - actual_len, 2)
            sliced = torch.cat([sliced, pad], dim=1)

        return sliced
