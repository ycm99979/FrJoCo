"""
LIPM MPC — Kajita Preview Control 방식 QP (ReLUQP 기반)

C++ FrMoCo controller/LIPM_MPC.cpp 대응:
  - LIPM 상태공간에서 prediction matrices (Mx, Mu) 빌드
  - H = blkdiag(alpha*I + gamma*Mu'Mu) 사전 계산 (오프라인)
  - 매 스텝: gradient 업데이트 → ReLUQP solve → 첫 jerk 반환

x/y 통합 QP:
  min  1/2 U' H U + g' U
  s.t. (unconstrained → dummy bounds)
  U = [jerk_x(0..N-1), jerk_y(0..N-1)]  (2N,)
"""

import torch
from pytorch.config.g1_config import WalkingConfig
from pytorch.dynamics_model.LIPM_dynamics import LIPMdynamics
from pytorch.utils.qp_solver import BatchQPSolver


class LIPM_MPC:
    """Kajita Preview MPC — ReLUQP 기반 배치 PyTorch 구현.

    C++ LIPM_MPC 클래스 대응.
    """

    def __init__(self, cfg: WalkingConfig):
        self.cfg = cfg
        self.N = cfg.mpc_horizon
        self.alpha = cfg.mpc_alpha
        self.gamma = cfg.mpc_gamma
        self.device = cfg.device
        self.dtype = cfg.dtype

        # LIPM 모델에서 A, B, C 가져옴 (base 행렬)
        lipm = LIPMdynamics(cfg)
        self.A_lipm = lipm.Ad[0]  # (3, 3)
        self.B_lipm = lipm.Bd[0]  # (3, 1)
        self.C_lipm = lipm.Cd[0]  # (1, 3)

        self._build_prediction_matrices()
        self._setup_qp()

    def _build_prediction_matrices(self):
        """Prediction matrices Mx, Mu 빌드.

        C++ buildPredictionMatrices() 대응.
        Mx: (N, 3) — ZMP prediction from state
        Mu: (N, N) — ZMP prediction from input 
        """
        A, B, C = self.A_lipm, self.B_lipm, self.C_lipm
        N = self.N
        device = self.device
        dtype = self.dtype

        Px = torch.zeros(3 * N, 3, device=device, dtype=dtype)
        Pu = torch.zeros(3 * N, N, device=device, dtype=dtype)

        A_pow = A.clone()
        for i in range(N):
            Px[3 * i:3 * (i + 1), :] = A_pow
            col_val = B.clone()
            for j in range(i, -1, -1):
                Pu[3 * i:3 * (i + 1), j:j + 1] = col_val
                col_val = A @ col_val
            A_pow = A @ A_pow

        Mx = torch.zeros(N, 3, device=device, dtype=dtype)
        Mu = torch.zeros(N, N, device=device, dtype=dtype)
        for i in range(N):
            Mx[i:i + 1, :] = C @ Px[3 * i:3 * (i + 1), :]
            for j in range(i + 1):
                Mu[i, j] = (C @ Pu[3 * i:3 * (i + 1), j:j + 1]).item()

        self.Mx = Mx  # (N, 3)
        self.Mu = Mu  # (N, N)

    def _setup_qp(self):
        """QP Hessian 빌드 + BatchQPSolver 초기화.

        H = blkdiag(H_single, H_single), H_single = alpha*I + gamma*Mu'Mu
        Unconstrained → A = I, l = -inf, u = +inf
        """
        N = self.N
        device = self.device
        dtype = self.dtype

        H_single = self.alpha * torch.eye(N, device=device, dtype=dtype) \
                   + self.gamma * self.Mu.T @ self.Mu

        H = torch.zeros(2 * N, 2 * N, device=device, dtype=dtype)
        H[:N, :N] = H_single
        H[N:, N:] = H_single
        self.H = H

        # Unconstrained: A = I, l = -inf, u = +inf
        n_var = 2 * N
        A_con = torch.eye(n_var, device=device, dtype=dtype)
        l_con = torch.full((n_var,), -1e10, device=device, dtype=dtype)
        u_con = torch.full((n_var,), 1e10, device=device, dtype=dtype)

        self.qp_solver = BatchQPSolver(
            H=H, A=A_con, l=l_con, u=u_con,
            device=device, dtype=dtype,
            max_iter=200,
            eps_abs=1e-4,
            warm_starting=True,
        )

    def solve(
        self,
        x0: torch.Tensor,        # (B, 3) — [x, dx, ddx]
        y0: torch.Tensor,        # (B, 3) — [y, dy, ddy]
        zmp_ref_x: torch.Tensor,  # (B, N) — 목표 ZMP x
        zmp_ref_y: torch.Tensor,  # (B, N) — 목표 ZMP y
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MPC 풀이 → 첫 jerk 반환.

        C++ solve() 대응.
        grad = 2*gamma * Mu' * (Mx*x0 - zmp_ref) per axis
        g_qp = [grad_x; grad_y]

        Returns:
            jerk_x: (B,) — receding horizon 첫 번째 x jerk
            jerk_y: (B,) — receding horizon 첫 번째 y jerk
        """
        Mx, Mu = self.Mx, self.Mu
        N = self.N
        gamma = self.gamma
        B = x0.shape[0]

        # residual: (N, B) → (B, N)
        res_x = ((Mx @ x0.T) - zmp_ref_x.T).T  # (B, N)
        res_y = ((Mx @ y0.T) - zmp_ref_y.T).T  # (B, N)

        # grad: (B, N)
        grad_x = 2.0 * gamma * (res_x @ Mu)  # (B, N) — Mu' @ res → res @ Mu (transposed)
        grad_y = 2.0 * gamma * (res_y @ Mu)

        # g_qp: (B, 2N)
        g_qp = torch.cat([grad_x, grad_y], dim=-1)  # (B, 2N)

        # QP solve
        U = self.qp_solver.solve(g_qp)  # (B, 2N)

        jerk_x = U[:, 0]   # (B,)
        jerk_y = U[:, N]   # (B,)

        return jerk_x, jerk_y
