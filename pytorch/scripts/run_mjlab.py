"""
run_mjlab.py — mjlab 기반 G1 보행 시뮬레이션.

C++ main.cpp 대응. Isaac Lab의 run_issac_lab.py와 동일 구조.
mjlab Simulation + Scene을 직접 사용하여 물리 루프 제어.

구조:
    1. G1MjLabEnv 초기화 (Simulation + Scene + Entity)
    2. G1MjLabPipeline 초기화 (오프라인 궤적 생성)
    3. 메인 루프:
        - 상태 수집 (get_state)
        - 물리 데이터 계산 (get_jacobians, get_mass_matrix, get_nle, get_com_jacobian)
        - MPC (mpc_decimation 주기)
        - WBC (매 스텝)
        - 토크 적용 → sim.step()

실행:
    python -m pytorch.scripts.run_mjlab --mode ik    (IK feedback + ForceOpt feedforward)
    python -m pytorch.scripts.run_mjlab --mode dbfc  (DBFC force control)
"""

from __future__ import annotations

import argparse
import sys
import os
import time

# 프로젝트 루트를 sys.path에 추가
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from pytorch.config.g1_config import WalkingConfig
from pytorch.main_controller.g1_mj_lab_pipeline import G1MjLabPipeline
from pytorch.sim.mj_lab_env import G1MjLabEnv


def parse_args():
    parser = argparse.ArgumentParser(description="G1 WBC — mjlab")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--mode", type=str, default="ik", choices=["ik", "dbfc"],
                        help="WBC 모드: ik (IK(fb)+ForceOpt(ff)) 또는 dbfc (DBFC 힘 제어)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--standing_time", type=float, default=2.0,
                        help="초기 standing 시간 (초)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="최대 스텝 수 (0=궤적 끝까지)")
    parser.add_argument("--render", action="store_true",
                        help="MuJoCo passive viewer로 시각화")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # ── WalkingConfig ──
    cfg = WalkingConfig(
        batch_size=args.num_envs,
        device=torch.device(device),
        dtype=torch.float32,
    )

    # ── 환경 초기화 ──
    print("=" * 70)
    print("[INIT] Creating mjlab environment...")
    env = G1MjLabEnv(
        walking_cfg=cfg,
        num_envs=args.num_envs,
        device=device,
    )
    env.reset()
    env.forward()

    # ── 초기 상태 ──
    state = env.get_state()

    # 시뮬레이터에서 실제 물리 파라미터 추출 → config 반영
    cfg.robot_mass = env.total_mass
    cfg.com_height = state['com_pos'][0, 2].item()

    print(f"  num_joints : {env.num_joints}")
    print(f"  num_bodies : {env.num_bodies}")
    print(f"  nq         : {env.nq}")
    print(f"  nv         : {env.nv}")
    print(f"  nu         : {env.nu}")
    print(f"  total_mass : {cfg.robot_mass:.4f} kg")
    print(f"  com_height : {cfg.com_height:.6f} m")
    print(f"  device     : {device}")
    print(f"  mode       : {args.mode}")

    print(f"  CoM        : {state['com_pos'][0].tolist()}")
    print(f"  LF         : {state['lf_pos'][0].tolist()}")
    print(f"  RF         : {state['rf_pos'][0].tolist()}")
    print(f"  lf_body    : {env.lf_body_idx} (global: {env.lf_body_global})")
    print(f"  rf_body    : {env.rf_body_idx} (global: {env.rf_body_global})")
    print(f"  MJ joints  : {env.mj_joint_names}")
    print(f"  Entity jnts: {env.entity_joint_names}")
    print("=" * 70)

    # ── Pipeline 초기화 (C++ 대응: nv=35, na=29) ──
    init_com = state["com_pos"].to(device=cfg.device, dtype=cfg.dtype)
    init_lf = state["lf_pos"].to(device=cfg.device, dtype=cfg.dtype)
    init_rf = state["rf_pos"].to(device=cfg.device, dtype=cfg.dtype)

    controller = G1MjLabPipeline(
        cfg, init_com, init_lf, init_rf,
        nv=env.nv, na=env.num_joints,  # C++: nv=35, na=29
    )
    traj_len = controller.foot_result['left_pos'].shape[1]
    print(f"[INIT] Pipeline ready — traj len: {traj_len} steps")

    use_dbfc = (args.mode == "dbfc")
    print(f"[INIT] WBC mode: {'DBFC (force control)' if use_dbfc else 'IK(fb) + ForceOpt+TorqueGen(ff)'}")
    print(f"[INIT] Standing phase: {args.standing_time}s, then walking")
    print(f"[INIT] MPC decimation: {cfg.mpc_decimation}")
    print("-" * 70)

    # ── Viewer (시각화) ──
    viewer = None
    if args.render:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(
            env._mj_model, env._mj_data,
            show_left_ui=False, show_right_ui=False,
        )
        print("[INIT] MuJoCo viewer launched")

    # ── 타이밍 ──
    mpc_decimation = cfg.mpc_decimation
    step_cnt = 0
    sim_time = 0.0
    standing_time = args.standing_time
    max_walk_time = traj_len * cfg.dt

    # 성능 측정
    mpc_times = []
    wbc_times = []

    # ── 메인 루프 ──
    try:
        while True:
            # viewer 닫히면 종료
            if viewer is not None and not viewer.is_running():
                print("\n[EXIT] Viewer closed")
                break
            # 최대 스텝 체크
            if args.max_steps > 0 and step_cnt >= args.max_steps:
                print(f"\n[DONE] max_steps={args.max_steps} 도달")
                break

            # 상태 수집
            state = env.get_state()
            com_pos = state["com_pos"]
            com_vel = state["com_vel"]

            # NaN 감지
            if torch.isnan(com_pos).any() or torch.isnan(state["joint_pos"]).any():
                print(f"\n[ERROR] step={step_cnt} NaN detected — 중단")
                break

            # 물리 데이터 (CPU MuJoCo → torch)
            jacobians = env.get_jacobians()
            mass_matrix = env.get_mass_matrix()
            nle = env.get_nle()
            com_jac = env.get_com_jacobian()

            if sim_time < standing_time:
                # ── Standing Phase ──
                tau = controller.standing_loop(
                    jacobians=jacobians,
                    mass_matrix=mass_matrix,
                    gravity_comp=nle,
                    joint_pos=state["joint_pos"],
                    joint_vel=state["joint_vel"],
                    com_pos=com_pos,
                    com_vel=com_vel,
                    lf_body_idx=env.lf_body_global,
                    rf_body_idx=env.rf_body_global,
                    com_jac=com_jac,
                    q_full=state["q_full"],
                    dq_full=state["dq_full"],
                    rf_pos_curr=state["rf_pos"],
                    lf_pos_curr=state["lf_pos"],
                    rf_ori_curr=state["rf_ori"],
                    lf_ori_curr=state["lf_ori"],
                )
            else:
                # ── Walking Phase ──
                walk_time = sim_time - standing_time

                # 궤적 끝 감지
                if walk_time >= max_walk_time - 0.1:
                    print(f"\n[DONE] 궤적 종료 — step={step_cnt}, t={sim_time:.3f}s")
                    break

                # MPC (매 mpc_decimation 스텝)
                if step_cnt % mpc_decimation == 0:
                    t0 = time.time()
                    controller.mpc_loop(com_pos, com_vel, walk_time)
                    mpc_times.append(time.time() - t0)

                # WBC (매 스텝)
                t0 = time.time()
                tau = controller.wbc_loop(
                    jacobians=jacobians,
                    mass_matrix=mass_matrix,
                    gravity_comp=nle,
                    joint_pos=state["joint_pos"],
                    joint_vel=state["joint_vel"],
                    com_pos=com_pos,
                    com_vel=com_vel,
                    lf_body_idx=env.lf_body_global,
                    rf_body_idx=env.rf_body_global,
                    sim_time=walk_time,
                    com_jac=com_jac,
                    q_full=state["q_full"],
                    dq_full=state["dq_full"],
                    rf_pos_curr=state["rf_pos"],
                    lf_pos_curr=state["lf_pos"],
                    rf_ori_curr=state["rf_ori"],
                    lf_ori_curr=state["lf_ori"],
                    use_dbfc=use_dbfc,
                )
                wbc_times.append(time.time() - t0)

            # 토크 적용 — tau: (B, na=29), 전체 관절 순서
            # leg_joint_ids = 다리12 + 허리3 = 15개 (entity 기준 0~14)
            # tau[:, 0:15]가 다리/허리 토크에 해당
            leg_tau = tau[:, :len(env.leg_joint_ids)].clamp(-300.0, 300.0)
            env.apply_torque(leg_tau)

            # sim step
            env.step()
            step_cnt += 1
            sim_time += cfg.dt

            # viewer sync
            if viewer is not None:
                env._sync_cpu_data(0)
                viewer.sync()

            # 주기적 출력
            if step_cnt % 100 == 0:
                com = com_pos[0]
                print(
                    f"[step={step_cnt:6d} t={sim_time:7.3f}s] "
                    f"CoM=({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:.4f}) "
                    f"|τ|={tau[0].norm():.2f} max={tau[0].abs().max():.2f}"
                )

            if step_cnt % 1000 == 0:
                avg_mpc = sum(mpc_times[-100:]) / max(len(mpc_times[-100:]), 1) * 1000
                avg_wbc = sum(wbc_times[-100:]) / max(len(wbc_times[-100:]), 1) * 1000
                print(f"  Timing: MPC avg={avg_mpc:.3f}ms  WBC avg={avg_wbc:.3f}ms")

                # 이상 감지
                if com_pos[0, 2] < 0.3:
                    print("  WARNING: CoM height < 0.3m")
                if torch.isnan(tau).any():
                    print("  ERROR: NaN in torque!")
                    break

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")

    # ── 종료 요약 ──
    if viewer is not None and viewer.is_running():
        viewer.close()
    print("=" * 70)
    print(f"[SUMMARY] {step_cnt} steps, {sim_time:.3f}s simulated")
    if mpc_times:
        print(f"  MPC: avg={sum(mpc_times)/len(mpc_times)*1000:.3f}ms")
    if wbc_times:
        print(f"  WBC: avg={sum(wbc_times)/len(wbc_times)*1000:.3f}ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
