"""
G1 Humanoid Genesis 시뮬레이션 실행 스크립트

사용법:
  python scripts/run_genesis.py                          # 기본 (1 env, viewer)
  python scripts/run_genesis.py --num_envs 16 --no-vis   # 16 병렬, headless
  python scripts/run_genesis.py --num_envs 4 --mode pd   # 4 병렬, PD 제어
"""

import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.genesis_env import G1GenesisEnv


def run_standing(env: G1GenesisEnv, steps: int = 2000):
    """제자리 서기 — gravity compensation + PD."""
    env.set_pd_gains()
    obs = env.get_obs()
    q0 = obs['dof_pos'].clone()  # (B, num_dofs) 초기 관절 위치

    print(f"[Standing] {steps} steps, {env.num_envs} envs")
    for i in range(steps):
        env.step_position(q0)

        if i % 500 == 0:
            obs = env.get_obs()
            h = obs['base_pos'][:, 2].mean().item()
            print(f"  step {i:4d} | height={h:.3f}m")


def run_torque_zero(env: G1GenesisEnv, steps: int = 1000):
    """토크 0 — 자유 낙하 테스트."""
    B = env.num_envs
    zero_tau = torch.zeros(B, env.num_dofs, device=env.device)

    print(f"[Free fall] {steps} steps, {env.num_envs} envs")
    for i in range(steps):
        env.step_torque(zero_tau)

        if i % 200 == 0:
            h = env.get_base_height().mean().item()
            fallen = env.is_fallen().sum().item()
            print(f"  step {i:4d} | height={h:.3f}m | fallen={fallen}/{B}")


def main():
    parser = argparse.ArgumentParser(description="G1 Genesis Simulation")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--mjcf", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "참고용", "mujoco", "g1_wbc", "g1", "scene_23dof.xml"),
                        help="MJCF 파일 경로 (절대 또는 상대)")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--mode", choices=["stand", "freefall"], default="stand")
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis")
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()

    env = G1GenesisEnv(
        num_envs=args.num_envs,
        mjcf_path=args.mjcf,
        dt=args.dt,
        show_viewer=args.vis,
        backend=args.backend,
    )

    print(f"\n=== G1 Genesis Env ===")
    print(f"  envs: {env.num_envs}")
    print(f"  dofs: {env.num_dofs}")
    print(f"  device: {env.device}")
    print(f"  dt: {env.dt}")

    obs = env.get_obs()
    print(f"\n[Initial state]")
    print(f"  base_pos: {obs['base_pos'][0].cpu().numpy()}")
    print(f"  base_quat: {obs['base_quat'][0].cpu().numpy()}")
    print(f"  dof_pos shape: {obs['dof_pos'].shape}")

    if args.mode == "stand":
        run_standing(env, args.steps)
    elif args.mode == "freefall":
        run_torque_zero(env, args.steps)

    print("\n[Done]")


if __name__ == "__main__":
    main()
