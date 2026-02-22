"""
ForceOptimizer — C++ Force_Optimizer.cpp 대응.

최적 지면 반력 F_hat 계산:
  min  (K*F - u)^T (K*F - u) + F^T W F
  s.t. l <= A_ineq * F <= u_ineq

  H = 2*(K^T K + W),  g = -2*K^T u

배치 PyTorch 구현. ReLUQP 기반 QP solver 사용.
"""

import torch
from pytorch.config.g1_config import WalkingConfig
from pytorch.utils.qp_solver import BatchQPSolver


class ForceOptimizer:
    """최적 지면 반력 QP — C++ ForceOptimizer 대응.

    num_vars = 12 (양발 × [fx, fy, fz, tx, ty, tz])
    num_ineq = 18 (마찰콘 5×2 + CoP 4×2)
    """

    def __init__(self, cfg: WalkingConfig, num_vars: int = 12, num_ineq: int = 18):
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.nv = num_vars
        self.n_ineq = num_ineq
        self.reg = cfg.force_opt_reg

        # 최적화 결과 버퍼
        B = cfg.batch_size
        self.opt_F = torch.zeros(B, num_vars, device=cfg.device, dtype=cfg.dtype)

        # QP solver는 H, A가 바뀌므로 매번 재구성
        # (ForceOptimizer는 H가 매 스텝 변하므로 BatchQPSolver 캐싱 불가)
        self._solver = None

    def solve(
        self,
        K: torch.Tensor,        # (B, 6, 12) — CoM dynamics 행렬
        u_vec: torch.Tensor,     # (B, 6) — 목표 wrench
        A_ineq: torch.Tensor,    # (B, n_ineq, 12) — 부등식 제약 행렬
        l_ineq: torch.Tensor,    # (B, n_ineq) — 하한
        u_ineq: torch.Tensor,    # (B, n_ineq) — 상한
    ) -> torch.Tensor:
        """QP 풀이 → 최적 반력 F_hat 반환.

        Returns:
            opt_F: (B, 12)
        """
        B = K.shape[0]
        nv = self.nv
        device = self.device
        dtype = self.dtype

        # W = reg * I
        W = self.reg * torch.eye(nv, device=device, dtype=dtype)  # (12, 12)

        # inf → 큰 값으로 치환 (ReLUQP는 inf 미지원)
        BIG = 1e6
        l_ineq = l_ineq.clamp(min=-BIG)
        u_ineq = u_ineq.clamp(max=BIG)

        # 배치별 QP 풀이 (H가 매번 바뀌므로 순차)
        for i in range(B):
            Ki = K[i]  # (6, 12)
            ui = u_vec[i]  # (6,)

            # H = 2*(K^T K + W), g = -2*K^T u
            H = 2.0 * (Ki.T @ Ki + W)
            g = -2.0 * (Ki.T @ ui)

            Ai = A_ineq[i]  # (n_ineq, 12)
            li = l_ineq[i]  # (n_ineq,)
            ui_con = u_ineq[i]  # (n_ineq,)

            # 단일 QP solver
            from pytorch.utils.qp_solver import solve_qp
            self.opt_F[i] = solve_qp(
                H, g, Ai, li, ui_con,
                device=device,
                max_iter=1000,
                eps_abs=1e-4,
            )

        return self.opt_F
