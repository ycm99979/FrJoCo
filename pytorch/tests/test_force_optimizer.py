"""
ForceOptimizer QP 테스트.

시나리오: 양발 접촉 상태에서 CoM 가속도 0 (정지 서기)
  → 최적 반력 F_hat이 중력 보상을 양발에 분배하는지 확인.

실행: python -m pytorch.tests.test_force_optimizer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from pytorch.config.g1_config import WalkingConfig
from pytorch.dynamics_model.com_dynamics import CenterOfMassDynamics
from pytorch.constraints.friction_cone import BatchedFrictionCone
from pytorch.constraints.cop_limits import BatchedCoPLimits
from pytorch.utils.qp_solver import solve_qp


def test_force_optimizer_standing():
    """양발 접촉 정지 서기 — 중력 보상 분배 테스트."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = WalkingConfig(batch_size=1, device=device, dtype=torch.float64)
    B = cfg.batch_size

    # ── 로봇 상태: 정지 서기 ──
    com_pos = torch.tensor([[0.035, 0.0, 0.69]], device=device, dtype=cfg.dtype)
    lf_pos = torch.tensor([[0.0, 0.1185, 0.0]], device=device, dtype=cfg.dtype)
    rf_pos = torch.tensor([[0.0, -0.1185, 0.0]], device=device, dtype=cfg.dtype)
    ddc_des = torch.zeros(B, 3, device=device, dtype=cfg.dtype)  # 가속도 0
    dL = torch.zeros(B, 3, device=device, dtype=cfg.dtype)

    # ── 1. CoM Dynamics → K, u ──
    com_dyn = CenterOfMassDynamics(cfg)
    K, u_vec = com_dyn.update(com_pos, lf_pos, rf_pos, ddc_des, dL)
    print(f"\nK shape: {K.shape}")  # (1, 6, 12)
    print(f"u shape: {u_vec.shape}")  # (1, 6)
    print(f"u (target wrench): {u_vec[0]}")

    # ── 2. 제약조건 조립 ──
    friction = BatchedFrictionCone(cfg)
    cop = BatchedCoPLimits(cfg, dX_max=0.1, dX_min=-0.05, dY_max=0.04, dY_min=-0.04)

    A_fric, l_fric, u_fric = friction.update()  # (B, 5, 3)
    contact = torch.ones(B, 2, device=device, dtype=cfg.dtype)  # 양발 접촉
    A_cop, l_cop, u_cop = cop.update(contact)  # (B, 8, 12)

    # 마찰콘을 12차원으로 확장
    A_fric_full = torch.zeros(B, 10, 12, device=device, dtype=cfg.dtype)
    A_fric_full[:, :5, :3] = A_fric      # 오른발 force [fx, fy, fz]
    A_fric_full[:, 5:, 6:9] = A_fric     # 왼발 force [fx, fy, fz]
    l_fric_full = torch.cat([l_fric, l_fric], dim=-1)  # (B, 10)
    u_fric_full = torch.cat([u_fric, u_fric], dim=-1)  # (B, 10)

    # 결합: (B, 18, 12)
    A_ineq = torch.cat([A_fric_full, A_cop], dim=1)
    l_ineq = torch.cat([l_fric_full, l_cop], dim=1)
    u_ineq = torch.cat([u_fric_full, u_cop], dim=1)

    print(f"\nA_ineq shape: {A_ineq.shape}")
    print(f"l_ineq shape: {l_ineq.shape}")
    print(f"u_ineq shape: {u_ineq.shape}")

    # ── inf → 큰 값으로 치환 (ReLUQP는 inf 미지원) ──
    BIG = 1e6
    l_ineq = l_ineq.clamp(min=-BIG)
    u_ineq = u_ineq.clamp(max=BIG)

    # ── 3. QP 구성 ──
    reg = cfg.force_opt_reg
    nv = 12
    W = reg * torch.eye(nv, device=device, dtype=cfg.dtype)

    Ki = K[0]       # (6, 12)
    ui = u_vec[0]   # (6,)
    H = 2.0 * (Ki.T @ Ki + W)
    g = -2.0 * (Ki.T @ ui)
    Ai = A_ineq[0]
    li = l_ineq[0]
    ui_con = u_ineq[0]

    print(f"\nH shape: {H.shape}")
    print(f"g shape: {g.shape}")
    print(f"H symmetric: {torch.allclose(H, H.T)}")
    print(f"H positive definite: {(torch.linalg.eigvalsh(H) > 0).all()}")
    print(f"H min eigenvalue: {torch.linalg.eigvalsh(H).min():.6e}")

    # ── 4. QP 풀이 ──
    print("\n--- Solving QP ---")
    F_hat = solve_qp(H, g, Ai, li, ui_con, device=device, max_iter=4000, eps_abs=1e-4)
    print(f"F_hat: {F_hat}")

    # ── 5. 검증 ──
    # F_hat = [fx_r, fy_r, fz_r, tx_r, ty_r, tz_r, fx_l, fy_l, fz_l, tx_l, ty_l, tz_l]
    fz_r = F_hat[2].item()
    fz_l = F_hat[8].item()
    total_fz = fz_r + fz_l
    expected_fz = cfg.robot_mass * cfg.gravity

    print(f"\n--- 검증 ---")
    print(f"오른발 fz: {fz_r:.4f} N")
    print(f"왼발  fz: {fz_l:.4f} N")
    print(f"합계  fz: {total_fz:.4f} N")
    print(f"기대값 (mg): {expected_fz:.4f} N")
    print(f"오차: {abs(total_fz - expected_fz):.6f} N")

    # 잔차 확인: K @ F - u
    residual = Ki @ F_hat - ui
    print(f"\n잔차 (K@F - u): {residual}")
    print(f"잔차 norm: {residual.norm():.6e}")

    # 마찰콘 확인: fz > 0
    print(f"\nfz_r > 0: {fz_r > 0}")
    print(f"fz_l > 0: {fz_l > 0}")

    # 마찰 제약 확인
    mu_eff = cfg.friction_mu / (2.0 ** 0.5)
    fx_r, fy_r = F_hat[0].item(), F_hat[1].item()
    fx_l, fy_l = F_hat[6].item(), F_hat[7].item()
    print(f"\n오른발 마찰: |fx|={abs(fx_r):.4f}, |fy|={abs(fy_r):.4f}, mu*fz={mu_eff*fz_r:.4f}")
    print(f"왼발  마찰: |fx|={abs(fx_l):.4f}, |fy|={abs(fy_l):.4f}, mu*fz={mu_eff*fz_l:.4f}")

    # 대칭성 확인 (CoM이 거의 중앙이므로 양발 힘이 비슷해야 함)
    print(f"\n양발 fz 비율: {fz_r / (fz_l + 1e-10):.4f} (1.0에 가까울수록 대칭)")

    print("\n✓ ForceOptimizer QP 테스트 완료")


if __name__ == "__main__":
    test_force_optimizer_standing()
