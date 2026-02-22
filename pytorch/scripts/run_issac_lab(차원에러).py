"""
run_issac_lab.py — C++ main.cpp 대응.

Isaac Lab bipeds.py 패턴: SimulationContext + Articulation 직접 사용.
DirectRLEnv 래퍼 없이 물리 루프를 직접 돌림.

구조:
    1. SimulationContext + Articulation 초기화
    2. G1WalkingPipeline 초기화 (오프라인 궤적 생성)
    3. 메인 루프:
        - MPC (mpc_decimation 주기)
        - WBC (매 스텝)
        - 토크 적용 → sim.step()

실행 (scripts 디렉토리에서):
    ./run.sh --num_envs 1 --mode ik
    ./run.sh --num_envs 1 --mode dbfc
"""

from __future__ import annotations
import sys
import os
import time

# 프로젝트 루트를 sys.path에 추가 (run.sh의 PYTHONPATH 보완)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 WBC — Isaac Lab")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--mode", type=str, default="ik", choices=["ik", "dbfc"],
                    help="WBC 모드: ik (IK+PD) 또는 dbfc (힘 제어)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 이후 import ──
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets import G1_CFG

from pytorch.config.g1_config import WalkingConfig
from pytorch.main_controller.g1_isaac_lab_pipeline import G1WalkingPipeline
from visualization.ground_reaction_force import create_contact_sensor, GRFLogger


# ── 디버그 로거 ──
class DebugLogger:
    """WBC 디버그 로거 — 주기적으로 상태/토크/에러 출력."""

    def __init__(self, print_every: int = 100, detail_every: int = 1000):
        self.print_every = print_every      # 간략 출력 주기 (스텝)
        self.detail_every = detail_every    # 상세 출력 주기 (스텝)
        self.mpc_time_acc = 0.0
        self.wbc_time_acc = 0.0
        self.mpc_count = 0
        self.wbc_count = 0
        self.tau_max_hist = []
        self.com_z_hist = []

    def log_mpc(self, elapsed: float, x_state, y_state, traj_idx: int):
        self.mpc_time_acc += elapsed
        self.mpc_count += 1

    def log_wbc(self, elapsed: float, tau, com_pos, com_des, lf_pos, rf_pos):
        self.wbc_time_acc += elapsed
        self.wbc_count += 1
        self.tau_max_hist.append(tau.abs().max().item())
        self.com_z_hist.append(com_pos[0, 2].item())

    def print_summary(self, step_cnt, sim_time, state, tau, controller):
        """간략 출력."""
        com = state["com_pos"][0]
        com_vel = state["com_vel"][0]
        lf = state["lf_pos"][0]
        rf = state["rf_pos"][0]

        print(
            f"[step={step_cnt:6d} t={sim_time:7.3f}s] "
            f"CoM=({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:.4f}) "
            f"vel=({com_vel[0]:+.3f}, {com_vel[1]:+.3f}, {com_vel[2]:+.3f}) "
            f"|τ|={tau[0].norm():.2f} max={tau[0].abs().max():.2f}"
        )

    def print_detail(self, step_cnt, sim_time, state, tau, controller, contact_sensor=None):
        """상세 출력."""
        com = state["com_pos"][0]
        com_vel = state["com_vel"][0]
        lf = state["lf_pos"][0]
        rf = state["rf_pos"][0]

        # MPC 상태
        x_st = controller.x_state[0]
        y_st = controller.y_state[0]

        # 접촉 상태
        contact = controller.get_contact_state(sim_time)

        # 평균 연산 시간
        avg_mpc = (self.mpc_time_acc / max(self.mpc_count, 1)) * 1000
        avg_wbc = (self.wbc_time_acc / max(self.wbc_count, 1)) * 1000

        # CoM 높이 통계
        if self.com_z_hist:
            z_min = min(self.com_z_hist[-100:])
            z_max = max(self.com_z_hist[-100:])
        else:
            z_min = z_max = com[2].item()

        # 토크 통계
        if self.tau_max_hist:
            tau_max_recent = max(self.tau_max_hist[-100:])
        else:
            tau_max_recent = 0.0

        print("=" * 70)
        print(f"[DETAIL step={step_cnt} t={sim_time:.3f}s]")
        print(f"  CoM pos : ({com[0]:+.5f}, {com[1]:+.5f}, {com[2]:.5f})")
        print(f"  CoM vel : ({com_vel[0]:+.5f}, {com_vel[1]:+.5f}, {com_vel[2]:+.5f})")
        print(f"  LF pos  : ({lf[0]:+.5f}, {lf[1]:+.5f}, {lf[2]:.5f})")
        print(f"  RF pos  : ({rf[0]:+.5f}, {rf[1]:+.5f}, {rf[2]:.5f})")
        print(f"  MPC x   : pos={x_st[0]:.5f} vel={x_st[1]:.5f} acc={x_st[2]:.5f}")
        print(f"  MPC y   : pos={y_st[0]:.5f} vel={y_st[1]:.5f} acc={y_st[2]:.5f}")
        print(f"  Contact : R={contact[0,0]:.0f} L={contact[0,1]:.0f}")
        print(f"  Traj idx: {controller.traj_idx}")
        print(f"  τ norm  : {tau[0].norm():.4f}  max={tau[0].abs().max():.4f}")
        print(f"  τ (legs): {tau[0, :12].tolist()}")
        print(f"  Timing  : MPC avg={avg_mpc:.3f}ms  WBC avg={avg_wbc:.3f}ms")
        print(f"  CoM z   : min={z_min:.5f} max={z_max:.5f} (last 100)")
        print(f"  τ max   : {tau_max_recent:.4f} (last 100)")

        # 이상 감지
        if com[2] < 0.3:
            print("  ⚠ WARNING: CoM height < 0.3m — 로봇 넘어짐 가능성")
        if com[2] > 1.0:
            print("  ⚠ WARNING: CoM height > 1.0m — 비정상 상승")
        if tau[0].abs().max() > 200:
            print("  ⚠ WARNING: 토크 > 200 Nm — 포화 가능성")
        if torch.isnan(tau).any():
            print("  ✗ ERROR: NaN in torque!")
        if torch.isnan(com).any():
            print("  ✗ ERROR: NaN in CoM!")

        print("=" * 70)

        # 누적 초기화
        self.mpc_time_acc = 0.0
        self.wbc_time_acc = 0.0
        self.mpc_count = 0
        self.wbc_count = 0
        self.tau_max_hist.clear()
        self.com_z_hist.clear()


def make_g1_cfg():
    """G1 ArticulationCfg — effort 제어용 (stiffness=0, damping=0)."""
    from isaaclab.actuators import ImplicitActuatorCfg

    cfg = G1_CFG.copy()
    cfg.actuators["legs"] = ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
            ".*_knee_joint", "torso_joint",
        ],
        effort_limit_sim=300, stiffness=0.0, damping=0.0,
        armature={".*_hip_.*": 0.01, ".*_knee_joint": 0.01, "torso_joint": 0.01},
    )
    cfg.actuators["feet"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit_sim=300, stiffness=0.0, damping=0.0, armature=0.01,
    )
    # 팔: 위치 제어 유지 (WBC 미제어, 초기 자세 유지)
    cfg.actuators["arms"] = ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint", ".*_elbow_pitch_joint",
            ".*_elbow_roll_joint",
            ".*_five_joint", ".*_three_joint", ".*_six_joint",
            ".*_four_joint", ".*_zero_joint", ".*_one_joint", ".*_two_joint",
        ],
        effort_limit_sim=300, stiffness=40.0, damping=10.0,
        armature=0.01,
    )
    cfg.prim_path = "/World/G1"

    # 초기 자세: C++ knees_bent (z=0.755, 무릎 굽힘)
    cfg.init_state.pos = (0.0, 0.0, 0.755)
    cfg.init_state.joint_pos = {
        "left_hip_pitch_joint": -0.312,
        "left_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363,
        "right_hip_pitch_joint": -0.312,
        "right_knee_joint": 0.669,
        "right_ankle_pitch_joint": -0.363,
        "torso_joint": 0.073,
    }
    return cfg


def design_scene(sim: SimulationContext):
    """씬 구성: 지면 + 조명 + G1 + 접촉 센서."""
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
    )
    robot = Articulation(make_g1_cfg())
    contact_sensor = create_contact_sensor("/World/G1")
    return robot, contact_sensor


def resolve_indices(robot: Articulation):
    """관절/바디 인덱스 매핑."""
    leg_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]
    waist_names = ["torso_joint"]
    arm_names = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
    ]

    leg_ids, _ = robot.find_joints(leg_names)
    waist_ids, _ = robot.find_joints(waist_names)
    arm_ids, _ = robot.find_joints(arm_names)
    ctrl_ids = list(leg_ids) + list(waist_ids)

    lf_ids, _ = robot.find_bodies("left_ankle_roll_link")
    rf_ids, _ = robot.find_bodies("right_ankle_roll_link")
    torso_ids, _ = robot.find_bodies("torso_link")

    return {
        "leg_ids": leg_ids,
        "waist_ids": waist_ids,
        "arm_ids": arm_ids,
        "ctrl_ids": ctrl_ids,
        "lf_body": lf_ids[0],
        "rf_body": rf_ids[0],
        "torso_body": torso_ids[0],
        "nv": 6 + robot.num_joints if not robot.is_fixed_base else robot.num_joints,
    }


def get_state(robot: Articulation, idx: dict) -> dict:
    """로봇 상태 수집."""
    d = robot.data
    dev = d.body_pos_w.device  # 시뮬레이션 device 기준
    mass = d.default_mass.to(dev)
    total_mass = mass.sum(-1, keepdim=True)
    com_pos = (mass.unsqueeze(-1) * d.body_com_pos_w).sum(1) / total_mass
    com_vel = (mass.unsqueeze(-1) * d.body_com_lin_vel_w).sum(1) / total_mass

    from isaaclab.utils.math import matrix_from_quat
    lf_quat = d.body_quat_w[:, idx["lf_body"], :]
    rf_quat = d.body_quat_w[:, idx["rf_body"], :]
    lf_ori = matrix_from_quat(lf_quat)
    rf_ori = matrix_from_quat(rf_quat)

    root_pos = d.root_pos_w
    root_quat = d.root_quat_w
    q_full = torch.cat([root_pos, root_quat, d.joint_pos], dim=-1)

    root_lin_vel = d.root_lin_vel_w
    root_ang_vel = d.root_ang_vel_w
    dq_full = torch.cat([root_lin_vel, root_ang_vel, d.joint_vel], dim=-1)

    return {
        "joint_pos": d.joint_pos,
        "joint_vel": d.joint_vel,
        "com_pos": com_pos,
        "com_vel": com_vel,
        "lf_pos": d.body_pos_w[:, idx["lf_body"], :],
        "rf_pos": d.body_pos_w[:, idx["rf_body"], :],
        "lf_ori": lf_ori,
        "rf_ori": rf_ori,
        "torso_pos": d.body_pos_w[:, idx["torso_body"], :],
        "q_full": q_full,
        "dq_full": dq_full,
    }


# ── CoM 자코비안 계산 ──
def compute_com_jacobian(robot: Articulation, idx: dict) -> torch.Tensor:
    """질량 가중 CoM 자코비안 (B, 3, nv)."""
    d = robot.data
    jacs = robot.root_physx_view.get_jacobians()    # (B, n_bodies, 6, nv)
    dev = jacs.device
    mass = d.default_mass.to(dev)                   # (B, n_bodies)
    total_mass = mass.sum(-1, keepdim=True)          # (B, 1)
    lin_jacs = jacs[:, :, :3, :]                     # (B, n_bodies, 3, nv)
    w = (mass / total_mass).unsqueeze(-1).unsqueeze(-1)  # (B, n_bodies, 1, 1)
    com_jac = (w * lin_jacs).sum(dim=1)              # (B, 3, nv)
    return com_jac


# ── main ──
def main():
    """메인 루프 — C++ main.cpp 대응."""

    # ── 시뮬레이션 설정 ──
    sim_cfg = sim_utils.SimulationCfg(dt=0.001, device=args_cli.device or "cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 0.5])

    robot, contact_sensor = design_scene(sim)
    sim.reset()
    robot.reset()
    contact_sensor.reset()

    # ── 초기 자세 설정 (C++ knees_bent 대응) ──
    idx = resolve_indices(robot)
    default_joint_pos = robot.data.default_joint_pos.clone()

    # 관절 이름 → 초기 각도 매핑 (C++ main.cpp 기준)
    knees_bent = {
        "left_hip_pitch_joint": -0.312,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.312,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.669,
        "right_ankle_pitch_joint": -0.363,
        "right_ankle_roll_joint": 0.0,
        "torso_joint": 0.073,
    }
    for jname, jval in knees_bent.items():
        jids, _ = robot.find_joints(jname)
        if jids:
            default_joint_pos[:, jids[0]] = jval

    robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos))
    sim.step()
    robot.update(sim_cfg.dt)
    state = get_state(robot, idx)

    # ── 초기 상태 출력 ──
    print("=" * 70)
    print("[INIT] Robot loaded")
    print(f"  num_joints : {robot.num_joints}")
    print(f"  num_bodies : {robot.num_bodies}")
    print(f"  nv (gen.)  : {idx['nv']}")
    print(f"  device     : {sim_cfg.device}")
    print(f"  mode       : {args_cli.mode}")
    print(f"  CoM        : {state['com_pos'][0].tolist()}")
    print(f"  LF         : {state['lf_pos'][0].tolist()}")
    print(f"  RF         : {state['rf_pos'][0].tolist()}")
    print(f"  lf_body_idx: {idx['lf_body']}")
    print(f"  rf_body_idx: {idx['rf_body']}")
    print("=" * 70)

    # ── WalkingConfig ──
    dev = torch.device(sim_cfg.device)
    cfg = WalkingConfig(
        batch_size=args_cli.num_envs,
        device=dev,
        dtype=torch.float32,
    )

    # ── Pipeline 초기화 ──
    init_com = state["com_pos"].to(device=dev, dtype=cfg.dtype)
    init_lf = state["lf_pos"].to(device=dev, dtype=cfg.dtype)
    init_rf = state["rf_pos"].to(device=dev, dtype=cfg.dtype)

    controller = G1WalkingPipeline(cfg, init_com, init_lf, init_rf,
                                    nv=idx["nv"], na=robot.num_joints)
    print(f"[INIT] Pipeline ready — traj len: {controller.foot_result['left_pos'].shape[1]} steps")

    use_dbfc = (args_cli.mode == "dbfc")
    print(f"[INIT] WBC mode: {'DBFC' if use_dbfc else 'IK+PD'}")

    # ── 디버그 로거 ──
    logger = DebugLogger(print_every=100, detail_every=1000)
    grf_logger = GRFLogger()

    # ── 타이밍 ──
    mpc_decimation = cfg.mpc_decimation
    step_cnt = 0
    sim_time = 0.0
    standing_time = 2.0  # 초기 standing 시간 (초)

    print(f"[INIT] Standing phase: {standing_time}s, then walking")
    print(f"[INIT] MPC decimation: {mpc_decimation} (MPC every {mpc_decimation} sim steps)")
    print("-" * 70)

    # ── 메인 루프 ──
    while simulation_app.is_running():
        # 상태 수집
        state = get_state(robot, idx)

        com_pos = state["com_pos"]
        com_vel = state["com_vel"]

        # NaN 감지
        if torch.isnan(com_pos).any() or torch.isnan(state["joint_pos"]).any():
            print(f"\n✗ [step={step_cnt}] NaN detected — 시뮬레이션 중단")
            print(f"  com_pos: {com_pos[0].tolist()}")
            print(f"  joint_pos has NaN: {torch.isnan(state['joint_pos']).any().item()}")
            break

        # Isaac Lab 물리 데이터
        jacobians = robot.root_physx_view.get_jacobians()
        mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()
        gravity_comp = robot.root_physx_view.get_gravity_compensation_forces()
        com_jac = compute_com_jacobian(robot, idx)

        if sim_time < standing_time:
            # ── Standing Phase ──
            tau = controller.standing_loop(
                jacobians=jacobians,
                mass_matrix=mass_matrix,
                gravity_comp=gravity_comp,
                joint_pos=state["joint_pos"],
                joint_vel=state["joint_vel"],
                com_pos=com_pos,
                com_vel=com_vel,
                lf_body_idx=idx["lf_body"],
                rf_body_idx=idx["rf_body"],
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

            # MPC (매 mpc_decimation 스텝)
            if step_cnt % mpc_decimation == 0:
                t0 = time.time()
                controller.mpc_loop(com_pos, com_vel, walk_time)
                mpc_elapsed = time.time() - t0
                logger.log_mpc(mpc_elapsed, controller.x_state, controller.y_state, controller.traj_idx)

            # WBC (매 스텝)
            t0 = time.time()
            tau = controller.wbc_loop(
                jacobians=jacobians,
                mass_matrix=mass_matrix,
                gravity_comp=gravity_comp,
                joint_pos=state["joint_pos"],
                joint_vel=state["joint_vel"],
                com_pos=com_pos,
                com_vel=com_vel,
                lf_body_idx=idx["lf_body"],
                rf_body_idx=idx["rf_body"],
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
            wbc_elapsed = time.time() - t0

            com_des = torch.stack([
                controller.x_state[:, 0],
                controller.y_state[:, 0],
                torch.full((cfg.batch_size,), cfg.com_height, device=cfg.device, dtype=cfg.dtype),
            ], dim=-1)
            logger.log_wbc(wbc_elapsed, tau, com_pos, com_des, state["lf_pos"], state["rf_pos"])

        # 토크 적용 — ctrl_ids 인덱스의 토크만 적용
        effort = torch.zeros_like(state["joint_pos"])
        ctrl_ids = idx["ctrl_ids"]
        for i, jid in enumerate(ctrl_ids):
            if jid < tau.shape[1]:
                effort[:, jid] = tau[:, jid]
        robot.set_joint_effort_target(effort)

        # sim step
        sim.step()
        robot.update(sim_cfg.dt)
        contact_sensor.update(sim_cfg.dt)
        grf_logger.update(contact_sensor)

        step_cnt += 1
        sim_time += sim_cfg.dt

        # 주기적 출력
        if step_cnt % logger.print_every == 0:
            logger.print_summary(step_cnt, sim_time, state, tau, controller)
            grf_logger.print_summary()

        if step_cnt % logger.detail_every == 0:
            logger.print_detail(step_cnt, sim_time, state, tau, controller, contact_sensor)
            grf_logger.print_detail(contact_sensor)

        # 궤적 끝 감지
        if sim_time > standing_time:
            walk_time = sim_time - standing_time
            max_time = controller.foot_result['left_pos'].shape[1] * cfg.dt
            if walk_time >= max_time - 0.1:
                print(f"\n[DONE] 궤적 종료 — step={step_cnt}, t={sim_time:.3f}s")
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C — 종료")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
