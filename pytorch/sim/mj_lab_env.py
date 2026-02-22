"""
MjLab G1 환경 래퍼 — C++ main.cpp 대응.

C++ 쪽 모델(model/g1/scene_29dof.xml)을 직접 사용.
29DOF, 모든 관절 motor(effort) 제어, Pinocchio 없이 MuJoCo CPU API로 동역학 계산.

C++ 구조:
  - MuJoCo: scene_29dof.xml (nq=36, nv=35, nu=29)
  - Pinocchio: g1_29dof.urdf → jacobian, mass_matrix, NLE
  - 모든 actuator가 <motor> (effort 직접 전달)

pytorch 대응:
  - MuJoCo CPU API로 jacobian/mass_matrix/NLE 계산 (Pinocchio 대체)
  - mjlab Simulation + Scene으로 GPU 물리 가속
  - C++과 동일한 관절 순서/인덱스 유지
"""

from __future__ import annotations

import os
import mujoco
import numpy as np
import torch

from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.terrains import TerrainImporterCfg

from pytorch.config.g1_config import WalkingConfig


# ──────────────────────────────────────────────────────────────
# C++ 모델 경로 (model/g1/scene_29dof.xml)
# ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_G1_29DOF_XML = os.path.join(_PROJECT_ROOT, "model", "g1", "g1_29dof.xml")


def _get_g1_29dof_spec() -> mujoco.MjSpec:
    """C++ 쪽 g1_29dof.xml을 MjSpec으로 로드."""
    assert os.path.exists(_G1_29DOF_XML), f"G1 29DOF XML not found: {_G1_29DOF_XML}"
    return mujoco.MjSpec.from_file(_G1_29DOF_XML)


# ──────────────────────────────────────────────────────────────
# C++ 모델 기준 EntityCfg
# ──────────────────────────────────────────────────────────────

# C++ g1_29dof.xml: 모든 29개 관절이 <motor> actuator
# 관절 순서 (C++ main.cpp qpos 인덱스 기준):
#   다리: L_hip_pitch(0), L_hip_roll(1), L_hip_yaw(2), L_knee(3),
#         L_ankle_pitch(4), L_ankle_roll(5),
#         R_hip_pitch(6), R_hip_roll(7), R_hip_yaw(8), R_knee(9),
#         R_ankle_pitch(10), R_ankle_roll(11)
#   허리: waist_yaw(12), waist_roll(13), waist_pitch(14)
#   팔:  L_shoulder_pitch(15)~L_wrist_yaw(21),
#         R_shoulder_pitch(22)~R_wrist_yaw(28)

# 모든 관절을 motor로 제어 (C++ 대응)
_ALL_MOTOR = BuiltinMotorActuatorCfg(
    target_names_expr=(
        # 다리 12
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        # 허리 3
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        # 팔 14
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ),
    effort_limit=300.0,
)

# C++ main.cpp knees_bent 초기 자세
_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.755),
    joint_pos={
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
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.073,
    },
    joint_vel={".*": 0.0},
)


def get_g1_29dof_cfg() -> EntityCfg:
    """C++ g1_29dof 모델 기준 EntityCfg — 모든 관절 motor 제어."""
    return EntityCfg(
        init_state=_INIT_STATE,
        spec_fn=_get_g1_29dof_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(_ALL_MOTOR,),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


# ──────────────────────────────────────────────────────────────
# MjLab G1 Environment — C++ main.cpp 대응
# ──────────────────────────────────────────────────────────────

class G1MjLabEnv:
    """mjlab 기반 G1 환경 — C++ main.cpp 구조 대응.

    C++ 모델(g1_29dof.xml) 사용:
      nq=36 (7 freejoint + 29 joints)
      nv=35 (6 floating base + 29 joints)
      nu=29 (모든 관절 motor)
      na=29 (actuated = 전체 관절)
    """

    def __init__(
        self,
        walking_cfg: WalkingConfig,
        num_envs: int = 1,
        device: str = "cuda:0",
    ):
        self.walking_cfg = walking_cfg
        self.num_envs = num_envs
        self.device = device

        # ── Scene 구성 (C++ 모델 사용) ──
        entity_cfg = get_g1_29dof_cfg()
        scene_cfg = SceneCfg(
            num_envs=num_envs,
            terrain=TerrainImporterCfg(terrain_type="plane"),
            entities={"robot": entity_cfg},
        )
        self.scene = Scene(scene_cfg, device=device)

        # ── Simulation 구성 ──
        sim_cfg = SimulationCfg(
            nconmax=45,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=walking_cfg.dt,
                integrator="implicitfast",
                solver="newton",
                iterations=10,
                ls_iterations=20,
            ),
        )
        mj_model = self.scene.compile()
        self.sim = Simulation(
            num_envs=num_envs,
            cfg=sim_cfg,
            model=mj_model,
            device=device,
        )

        # ── Scene 초기화 ──
        self.scene.initialize(
            mj_model=self.sim.mj_model,
            model=self.sim.model,
            data=self.sim.data,
        )

        # ── 인덱스 매핑 ──
        self.robot: Entity = self.scene["robot"]
        self._resolve_indices()

        # ── CPU MuJoCo 모델/데이터 (jacobian, mass matrix 계산용) ──
        self._mj_model = self.sim.mj_model
        self._mj_data = mujoco.MjData(self._mj_model)

        # 차원 정보 (C++ 대응: nq=36, nv=35, nu=29)
        self.nq = self._mj_model.nq
        self.nv = self._mj_model.nv
        self.nu = self._mj_model.nu
        self.nbody = self._mj_model.nbody

        # 총 질량 (C++ pinocchio::computeTotalMass 대응)
        self.total_mass = float(sum(self._mj_model.body_mass))

        # MuJoCo 모델 기준 관절 이름 순서 (디버그용)
        self.mj_joint_names = [
            mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self._mj_model.njnt)
        ]

    def _resolve_indices(self):
        """관절/바디 인덱스 매핑 — C++ resolve_indices 대응."""
        robot = self.robot

        # 바디 인덱스 (entity-local)
        lf_ids, _ = robot.find_bodies("left_ankle_roll_link")
        rf_ids, _ = robot.find_bodies("right_ankle_roll_link")
        torso_ids, _ = robot.find_bodies("torso_link")
        pelvis_ids, _ = robot.find_bodies("pelvis")

        self.lf_body_idx = lf_ids[0]
        self.rf_body_idx = rf_ids[0]
        self.torso_body_idx = torso_ids[0]
        self.pelvis_body_idx = pelvis_ids[0] if pelvis_ids else 0

        # 글로벌 바디 인덱스 (mujoco 모델 기준 — C++ frame_id 대응)
        self.lf_body_global = self.robot.indexing.body_ids[self.lf_body_idx].item()
        self.rf_body_global = self.robot.indexing.body_ids[self.rf_body_idx].item()
        self.torso_body_global = self.robot.indexing.body_ids[self.torso_body_idx].item()

        # C++ 제어 관절: 다리12 + 허리3 = 15 (WBC 토크 적용 대상)
        ctrl_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        ]
        self.leg_joint_ids, self.leg_joint_names = robot.find_joints(ctrl_names)

        self.num_joints = robot.num_joints  # 29 (C++ 대응)
        self.num_bodies = robot.num_bodies

        # Entity 기준 관절 이름 순서
        self.entity_joint_names = list(robot.joint_names)

    def reset(self):
        """환경 리셋 — knees_bent 초기 자세 적용."""
        self.sim.reset()
        self.scene.reset()
        self.scene.write_data_to_sim()

        # mjlab scene.reset()이 joint_pos를 적용하지 않는 문제 우회:
        # CPU mj_data에 직접 초기 관절값 세팅 후 GPU qpos에 복사
        self._apply_init_pose()

        self.sim.forward()

    def _apply_init_pose(self):
        """초기 관절 자세를 GPU qpos에 직접 적용.

        mjlab EntityCfg.InitialStateCfg의 joint_pos가 GPU sim에 반영되지 않는
        문제를 우회. CPU mj_model 기준 관절 이름 → qpos 인덱스 매핑 후 직접 쓰기.
        관절 이름은 mjlab이 'robot/' prefix를 붙이므로 양쪽 모두 시도.
        """
        mj_model = self._mj_model
        # 관절 이름 → qpos 인덱스 매핑
        name_to_qadr: dict[str, int] = {}
        for i in range(mj_model.njnt):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                name_to_qadr[name] = int(mj_model.jnt_qposadr[i])

        # GPU qpos를 CPU로 가져와서 수정
        qpos_np = self.sim.data.qpos[0].detach().cpu().numpy().copy()

        for jname, jval in _INIT_STATE.joint_pos.items():
            # mjlab은 'robot/' prefix를 붙임
            for candidate in (f"robot/{jname}", jname):
                if candidate in name_to_qadr:
                    qpos_np[name_to_qadr[candidate]] = jval
                    break

        # 초기 높이도 적용
        qpos_np[2] = _INIT_STATE.pos[2]  # z = 0.755

        # GPU에 다시 쓰기 (모든 env에 동일하게 적용)
        import torch as _torch
        qpos_t = _torch.from_numpy(qpos_np).to(device=self.device, dtype=_torch.float32)
        for env_id in range(self.num_envs):
            self.sim.data.qpos[env_id, :len(qpos_np)] = qpos_t

    def step(self):
        """물리 1스텝 진행."""
        self.scene.write_data_to_sim()
        self.sim.step()

    def forward(self):
        """Forward kinematics 갱신."""
        self.sim.forward()

    # ── 상태 접근 ──

    def get_state(self) -> dict[str, torch.Tensor]:
        """로봇 상태 수집 — C++ Pinocchio FK 대응."""
        data = self.robot.data

        joint_pos = data.joint_pos  # (B, 29)
        joint_vel = data.joint_vel  # (B, 29)

        # CoM
        com_pos = self.sim.data.subtree_com[:, self.robot.indexing.root_body_id]
        root_vel = data.root_com_vel_w  # (B, 6) [lin, ang]
        com_vel = root_vel[:, :3]

        # 발 위치/자세
        lf_pos = data.body_link_pos_w[:, self.lf_body_idx]
        rf_pos = data.body_link_pos_w[:, self.rf_body_idx]
        lf_quat = data.body_link_quat_w[:, self.lf_body_idx]  # (B, 4) [w,x,y,z]
        rf_quat = data.body_link_quat_w[:, self.rf_body_idx]
        lf_ori = _quat_to_rotmat(lf_quat)
        rf_ori = _quat_to_rotmat(rf_quat)

        # 전체 일반화 좌표/속도 (C++ q_pin, dq 대응)
        # MuJoCo qpos: [pos(3), quat_wxyz(4), joints(29)] = 36
        # Pinocchio/IK 기대: [pos(3), quat_xyzw(4), joints(29)] = 36
        q_mj = self.sim.data.qpos[:, :self.nq]   # (B, 36)
        q_full = q_mj.clone()
        # MuJoCo wxyz → xyzw 변환 (C++ mj2pin_q 대응)
        q_full[:, 3] = q_mj[:, 4]  # x
        q_full[:, 4] = q_mj[:, 5]  # y
        q_full[:, 5] = q_mj[:, 6]  # z
        q_full[:, 6] = q_mj[:, 3]  # w
        dq_full = self.sim.data.qvel[:, :self.nv]  # (B, 35)

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "com_pos": com_pos,
            "com_vel": com_vel,
            "lf_pos": lf_pos,
            "rf_pos": rf_pos,
            "lf_ori": lf_ori,
            "rf_ori": rf_ori,
            "q_full": q_full,
            "dq_full": dq_full,
        }

    # ── 물리 데이터 (C++ Pinocchio 대응) ──

    def _sync_cpu_data(self, env_id: int = 0):
        """GPU 상태를 CPU MuJoCo 데이터로 동기화."""
        qpos = self.sim.data.qpos[env_id].detach().cpu().numpy()
        qvel = self.sim.data.qvel[env_id].detach().cpu().numpy()
        self._mj_data.qpos[:] = qpos[:self._mj_model.nq]
        self._mj_data.qvel[:] = qvel[:self._mj_model.nv]
        mujoco.mj_forward(self._mj_model, self._mj_data)

    def get_jacobians(self, env_id: int = 0) -> torch.Tensor:
        """Body jacobians — C++ pinocchio::computeJointJacobians 대응.

        Returns:
            jacobians: (1, nbody, 6, nv) — [angular(3), linear(3)]
        """
        self._sync_cpu_data(env_id)
        nv = self._mj_model.nv
        nbody = self._mj_model.nbody

        jac_all = np.zeros((nbody, 6, nv), dtype=np.float64)
        for b in range(nbody):
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            mujoco.mj_jacBody(self._mj_model, self._mj_data, jacp, jacr, b)
            jac_all[b, 0:3, :] = jacr  # angular
            jac_all[b, 3:6, :] = jacp  # linear

        return torch.from_numpy(jac_all).to(
            device=self.device, dtype=self.walking_cfg.dtype
        ).unsqueeze(0)

    def get_mass_matrix(self, env_id: int = 0) -> torch.Tensor:
        """Mass matrix — C++ pinocchio::crba 대응.

        Returns:
            M: (1, nv, nv) — nv=35
        """
        self._sync_cpu_data(env_id)
        nv = self._mj_model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(self._mj_model, M, self._mj_data.qM)

        return torch.from_numpy(M).to(
            device=self.device, dtype=self.walking_cfg.dtype
        ).unsqueeze(0)

    def get_nle(self, env_id: int = 0) -> torch.Tensor:
        """Nonlinear effects — C++ pinocchio::nonLinearEffects 대응.

        Returns:
            nle: (1, nv) — nv=35
        """
        self._sync_cpu_data(env_id)
        return torch.from_numpy(self._mj_data.qfrc_bias.copy()).to(
            device=self.device, dtype=self.walking_cfg.dtype
        ).unsqueeze(0)

    def get_com_jacobian(self, env_id: int = 0) -> torch.Tensor:
        """CoM jacobian — C++ pinocchio::jacobianCenterOfMass 대응.

        Returns:
            com_jac: (1, 3, nv) — nv=35
        """
        self._sync_cpu_data(env_id)
        nv = self._mj_model.nv
        nbody = self._mj_model.nbody

        total_mass = 0.0
        com_jac = np.zeros((3, nv))
        for b in range(1, nbody):
            mass_b = self._mj_model.body_mass[b]
            if mass_b < 1e-10:
                continue
            jacp = np.zeros((3, nv))
            mujoco.mj_jacBody(self._mj_model, self._mj_data, jacp, None, b)
            com_jac += mass_b * jacp
            total_mass += mass_b

        if total_mass > 0:
            com_jac /= total_mass

        return torch.from_numpy(com_jac).to(
            device=self.device, dtype=self.walking_cfg.dtype
        ).unsqueeze(0)

    # ── 토크 적용 ──

    def apply_torque(self, tau: torch.Tensor):
        """다리/허리 관절에 effort 토크 적용 — C++ g_d->ctrl 대응.

        Args:
            tau: (B, 15) — leg_joint_ids 순서 (다리12 + 허리3)
        """
        self.robot.set_joint_effort_target(
            effort=tau,
            joint_ids=self.leg_joint_ids,
        )


# ──────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────

def _quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) → rotation matrix (B, 3, 3)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    R = torch.zeros(*quat.shape[:-1], 3, 3, device=quat.device, dtype=quat.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R
