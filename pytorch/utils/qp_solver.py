"""
QP Solver 래퍼 — ReLUQP 기반

ReLUQP-py (GPU-Accelerated ADMM QP Solver) 래핑.
  - 단일 QP: solve_qp()
  - 배치 QP (동일 H, A / 다른 g, l, u): BatchQPSolver

QP 형태:
  min  1/2 x' H x + g' x
  s.t. l <= A x <= u

ReLUQP는 setup 시 H, A 기반 ADMM 행렬을 사전 계산하므로,
H, A가 동일한 반복 풀이에서 update(g, l, u)만으로 빠르게 재풀이 가능.
"""

import sys
import os
import torch

# ReLUQP-py를 import path에 추가
_reluqp_path = os.path.join(os.path.dirname(__file__), '..', '..', 'QP_solver_lists', 'ReLUQP-py')
if _reluqp_path not in sys.path:
    sys.path.insert(0, os.path.abspath(_reluqp_path))

from reluqp.reluqpth import ReLU_QP


def solve_qp(
    H: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    device: torch.device = None,
    max_iter: int = 4000,
    eps_abs: float = 1e-3,
) -> torch.Tensor:
    """단일 QP 풀이. 결과 x 반환."""
    if device is None:
        device = H.device
    solver = ReLU_QP()
    solver.setup(H, g, A, l, u, device=device, max_iter=max_iter, eps_abs=eps_abs)
    results = solver.solve()
    return results.x


class BatchQPSolver:
    """배치 QP Solver — 동일 H, A / 다른 g, l, u.

    LIPM MPC처럼 H, A가 고정이고 매 스텝 g만 바뀌는 경우에 최적.
    setup()에서 ADMM 행렬 사전 계산 → solve()에서 g/l/u만 업데이트.

    배치 처리: B개의 독립 QP를 순차 풀이.
    (ReLUQP가 단일 QP만 지원하므로, B가 작으면 충분히 빠름)
    """

    def __init__(
        self,
        H: torch.Tensor,       # (nx, nx)
        A: torch.Tensor,       # (nc, nx)
        l: torch.Tensor,       # (nc,) — 기본 하한
        u: torch.Tensor,       # (nc,) — 기본 상한
        device: torch.device = None,
        dtype: torch.dtype = None,
        max_iter: int = 4000,
        eps_abs: float = 1e-3,
        warm_starting: bool = True,
    ):
        self.nx = H.shape[0]
        self.nc = A.shape[0]
        self.device = device or H.device
        self.dtype = dtype or H.dtype
        self.max_iter = max_iter
        self.eps_abs = eps_abs
        self.warm_starting = warm_starting

        # 모든 텐서를 동일 device/dtype으로 이동
        H = H.to(device=self.device, dtype=self.dtype)
        A = A.to(device=self.device, dtype=self.dtype)
        l = l.to(device=self.device, dtype=self.dtype)
        u = u.to(device=self.device, dtype=self.dtype)

        # 기본 g (zero) 로 setup — ADMM 행렬은 H, A에만 의존
        g_dummy = torch.zeros(self.nx, device=self.device, dtype=self.dtype)
        self._solver = ReLU_QP()
        self._solver.setup(
            H, g_dummy, A, l, u,
            device=self.device,
            precision=self.dtype,
            max_iter=max_iter,
            eps_abs=eps_abs,
            warm_starting=warm_starting,
        )

        # H, A 저장 (배치 solver 재생성용)
        self.H = H
        self.A = A

    def solve(
        self,
        g: torch.Tensor,       # (B, nx) 또는 (nx,)
        l: torch.Tensor = None,  # (B, nc) 또는 (nc,) — None이면 setup 값 유지
        u: torch.Tensor = None,  # (B, nc) 또는 (nc,) — None이면 setup 값 유지
    ) -> torch.Tensor:          # (B, nx)
        """배치 QP 풀이.

        g가 1D면 단일 QP, 2D면 배치.
        """
        if g.dim() == 1:
            return self._solve_single(g, l, u).unsqueeze(0)

        B = g.shape[0]
        results = torch.zeros(B, self.nx, device=self.device, dtype=g.dtype)

        for i in range(B):
            gi = g[i]
            li = l[i] if l is not None and l.dim() == 2 else l
            ui = u[i] if u is not None and u.dim() == 2 else u
            results[i] = self._solve_single(gi, li, ui)

        return results

    def _solve_single(
        self,
        g: torch.Tensor,       # (nx,)
        l: torch.Tensor = None,  # (nc,) or None
        u: torch.Tensor = None,  # (nc,) or None
    ) -> torch.Tensor:
        """단일 QP 풀이 (update → solve)."""
        # ReLUQP update는 numpy를 기대하므로 텐서로 직접 설정
        self._solver.QP.g = g.to(device=self.device, dtype=self.dtype).contiguous()
        # b_ks 재계산 (g 변경 시 필요)
        for idx in range(len(self._solver.layers.rhos)):
            self._solver.layers.b_ks[idx] = (self._solver.layers.B_ks[idx] @ self._solver.QP.g).contiguous()

        if l is not None:
            self._solver.QP.l = l.to(device=self.device, dtype=self.dtype).contiguous()
        if u is not None:
            self._solver.QP.u = u.to(device=self.device, dtype=self.dtype).contiguous()

        if not self.warm_starting:
            self._solver.clear_primal_dual()

        result = self._solver.solve()
        return result.x.clone()
