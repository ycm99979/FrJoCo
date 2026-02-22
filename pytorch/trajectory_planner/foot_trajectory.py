"""
배치 Foot Trajectory — PyTorch 벡터화 (미분가능)

C++ FrMoCo foot_trajectory.cpp 대응:
  - XY: Cycloid 보간 (이착지 속도 0 자동 보장)
  - Z: 5차 Bezier (이착지 속도/가속도 0)
  - 반환: pos + vel + acc (FootTrajectoryResult 대응)

핵심 최적화:
  1. for 루프 최소화 → 전체 궤적을 (B, T, 3) 텐서로 한번에 생성
  2. Phase 분기를 torch.where 마스킹으로 처리 (GPU 분기 없음)
  3. batch_utils의 cycloid/bezier 함수 재사용
  4. 모든 연산이 autograd 호환 → MPC 파라미터 역전파 가능
"""

import torch
from pytorch.config.g1_config import WalkingConfig
from pytorch.utils.batch_utils import (
    cycloid_pos, cycloid_vel, cycloid_acc,
    bezier_z_pos, bezier_z_vel, bezier_z_acc,
)


class FootTrajectoryGenerator:
    """배치 발 궤적 생성기 — Cycloid XY + 5차 Bezier Z.

    C++ FootTrajectory::computeFull() 대응.
    반환: left/right × pos/vel/acc — 총 6개 텐서.
    """

    def __init__(self, cfg: WalkingConfig):
        self.cfg = cfg

    def generate(
        self,
        footsteps: torch.Tensor,    # (B, N, 2) footstep XY 위치
        init_lf: torch.Tensor,      # (B, 3) 초기 왼발 [x, y, z]
        init_rf: torch.Tensor,      # (B, 3) 초기 오른발 [x, y, z]
    ) -> dict[str, torch.Tensor]:
        """전체 발 궤적을 배치 텐서로 한번에 생성.

        C++ computeFull() 대응:
          - DSP 구간: 발 정지 (vel=acc=0)
          - SSP 구간: Cycloid XY + Bezier Z (pos/vel/acc)

        Returns:
            dict with keys:
              left_pos, right_pos:  (B, T, 3)
              left_vel, right_vel:  (B, T, 3)
              left_acc, right_acc:  (B, T, 3)
        """
        cfg = self.cfg
        B, N, _ = footsteps.shape
        device = footsteps.device
        dtype = footsteps.dtype
        S = cfg.samples_per_step
        T = N * S
        ssp = cfg.ssp_time
        h = cfg.step_height

        # ── 1. 시간 & 위상 텐서 ──
        step_local_t = torch.arange(S, device=device, dtype=dtype) * cfg.dt  # (S,)
        dsp_mask = step_local_t < cfg.dsp_time  # (S,)
        swing_phase = torch.clamp(
            (step_local_t - cfg.dsp_time) / ssp, 0.0, 1.0
        )  # (S,)

        # ── 2. 스텝별 swing 여부 ──
        step_idx = torch.arange(N, device=device)
        is_right_swing = (step_idx % 2 == 0)  # (N,) — C++: i%2==0 → right swing
        has_swing = torch.ones(N, device=device, dtype=torch.bool)
        has_swing[-1] = False  # 마지막 스텝은 swing 없음

        # ── 3. Swing target (다음 스텝 위치) ──
        swing_target_xy = torch.cat([
            footsteps[:, 1:, :],
            footsteps[:, -1:, :],
        ], dim=1)  # (B, N, 2)

        # ── 4. 스텝별 발 위치 누적 (N은 ~20이라 루프 OK) ──
        left_pos_xy = torch.zeros(B, N, 2, device=device, dtype=dtype)
        right_pos_xy = torch.zeros(B, N, 2, device=device, dtype=dtype)
        left_pos_xy[:, 0] = init_lf[:, :2]
        right_pos_xy[:, 0] = init_rf[:, :2]

        for i in range(1, N):
            if is_right_swing[i - 1]:
                left_pos_xy[:, i] = left_pos_xy[:, i - 1]
                right_pos_xy[:, i] = swing_target_xy[:, i - 1] if has_swing[i - 1] else right_pos_xy[:, i - 1]
            else:
                right_pos_xy[:, i] = right_pos_xy[:, i - 1]
                left_pos_xy[:, i] = swing_target_xy[:, i - 1] if has_swing[i - 1] else left_pos_xy[:, i - 1]

        gz_lf = init_lf[:, 2]  # (B,)
        gz_rf = init_rf[:, 2]  # (B,)

        # ── 5. Swing 변위 (dx, dy) — C++ dx_swing, dy_swing 대응 ──
        # swing하는 발의 현재 위치
        swing_start_xy = torch.where(
            is_right_swing.view(1, N, 1).expand(B, N, 2),
            right_pos_xy, left_pos_xy,
        )  # (B, N, 2)
        d_swing = swing_target_xy - swing_start_xy  # (B, N, 2) — [dx, dy]

        # ── 6. phase → cycloid/bezier 계수 (batch_utils 사용) ──
        p = swing_phase  # (S,)

        c_pos = cycloid_pos(p)          # (S,)
        c_vel = cycloid_vel(p, ssp)     # (S,)
        c_acc = cycloid_acc(p, ssp)     # (S,)

        # Z: ground_z per swing foot
        gz = torch.where(
            is_right_swing.view(1, N), gz_rf.view(B, 1), gz_lf.view(B, 1)
        )  # (B, N)

        # bezier: p=(S,), gz=(B,N) → expand to (B, N, S)
        p_exp = p.unsqueeze(0).unsqueeze(0).expand(B, N, S)       # (B, N, S)
        gz_exp = gz.unsqueeze(-1).expand(B, N, S)                  # (B, N, S)

        bz_pos = bezier_z_pos(p_exp, gz_exp, h)                   # (B, N, S)
        bz_vel = bezier_z_vel(p_exp, h, ssp)                      # (B, N, S)
        bz_acc = bezier_z_acc(p_exp, h, ssp)                      # (B, N, S)

        # ── 7. XY pos/vel/acc 조립 ──
        # d_swing: (B, N, 2) → (B, N, 1, 2)
        d_xy = d_swing.unsqueeze(2)                                # (B, N, 1, 2)
        # swing_start: (B, N, 2) → (B, N, 1, 2)
        start_xy = swing_start_xy.unsqueeze(2)                    # (B, N, 1, 2)

        # cycloid 계수: (S,) → (1, 1, S, 1)
        c_pos_4d = c_pos.view(1, 1, S, 1)
        c_vel_4d = c_vel.view(1, 1, S, 1)
        c_acc_4d = c_acc.view(1, 1, S, 1)

        swing_xy_pos = start_xy + d_xy * c_pos_4d                 # (B, N, S, 2)
        swing_xy_vel = d_xy * c_vel_4d                             # (B, N, S, 2)
        swing_xy_acc = d_xy * c_acc_4d                             # (B, N, S, 2)

        # ── 8. 마스킹: DSP 구간 + swing 없는 스텝 → 정지 ──
        active = (~dsp_mask.unsqueeze(0)) & has_swing.unsqueeze(-1)  # (N, S)
        active = active.unsqueeze(0).expand(B, N, S)                 # (B, N, S)
        rs_mask = is_right_swing.view(1, N, 1).expand(B, N, S)      # (B, N, S)

        # stance 위치
        left_stance_xy = left_pos_xy.unsqueeze(2).expand(B, N, S, 2)
        right_stance_xy = right_pos_xy.unsqueeze(2).expand(B, N, S, 2)
        left_stance_z = gz_lf.view(B, 1, 1).expand(B, N, S)
        right_stance_z = gz_rf.view(B, 1, 1).expand(B, N, S)
        zeros_xy = torch.zeros(B, N, S, 2, device=device, dtype=dtype)
        zeros_z = torch.zeros(B, N, S, device=device, dtype=dtype)

        # ── 9. 최종 조립 ──
        # 왼발
        lf_swing = active & ~rs_mask  # (B, N, S)
        left_xy_p = torch.where(lf_swing.unsqueeze(-1), swing_xy_pos, left_stance_xy)
        left_xy_v = torch.where(lf_swing.unsqueeze(-1), swing_xy_vel, zeros_xy)
        left_xy_a = torch.where(lf_swing.unsqueeze(-1), swing_xy_acc, zeros_xy)
        left_z_p = torch.where(lf_swing, bz_pos, left_stance_z)
        left_z_v = torch.where(lf_swing, bz_vel, zeros_z)
        left_z_a = torch.where(lf_swing, bz_acc, zeros_z)

        # 오른발
        rf_swing = active & rs_mask  # (B, N, S)
        right_xy_p = torch.where(rf_swing.unsqueeze(-1), swing_xy_pos, right_stance_xy)
        right_xy_v = torch.where(rf_swing.unsqueeze(-1), swing_xy_vel, zeros_xy)
        right_xy_a = torch.where(rf_swing.unsqueeze(-1), swing_xy_acc, zeros_xy)
        right_z_p = torch.where(rf_swing, bz_pos, right_stance_z)
        right_z_v = torch.where(rf_swing, bz_vel, zeros_z)
        right_z_a = torch.where(rf_swing, bz_acc, zeros_z)

        # (B, N, S, 3) → (B, T, 3)
        def assemble(xy, z):
            return torch.cat([xy, z.unsqueeze(-1)], dim=-1).reshape(B, T, 3)

        return {
            'left_pos':  assemble(left_xy_p, left_z_p),
            'right_pos': assemble(right_xy_p, right_z_p),
            'left_vel':  assemble(left_xy_v, left_z_v),
            'right_vel': assemble(right_xy_v, right_z_v),
            'left_acc':  assemble(left_xy_a, left_z_a),
            'right_acc': assemble(right_xy_a, right_z_a),
        }
