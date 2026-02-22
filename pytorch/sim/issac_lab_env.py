"""
Isaac Lab G1 환경 래퍼.

역할: 시뮬레이터 초기화 + 상태 제공 + 토크 적용.
컨트롤 로직은 run_issac_lab.py (main.cpp 역할) 에서 담당.
"""

from __future__ import annotations
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from pytorch.config.g1_config import WalkingConfig


# ──────────────────────────────────────────────────────────────
# ArticulationCfg
# ──────────────────────────────────────────────────────────────

def _make_g1_articulation_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),
            joint_pos={
                ".*_hip_pitch_joint":    -0.20,
                ".*_knee_joint":          0.42,
                ".*_ankle_pitch_joint":  -0.23,
                ".*_elbow_pitch_joint":   0.87,
                "left_shoulder_roll_joint":   0.16,
                "left_shoulder_pitch_joint":  0.35,
                "right_shoulder_roll_joint": -0.16,
                "right_shoulder_pitch_joint": 0.35,
                "left_one_joint":   1.0,  "right_one_joint": -1.0,
                "left_two_joint":   0.52, "right_two_joint": -0.52,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            # 다리/허리: stiffness=0, damping=0 → 코드에서 직접 토크 전달
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
                    ".*_knee_joint", "torso_joint",
                ],
                effort_limit_sim=300, stiffness=0.0, damping=0.0,
                armature={".*_hip_.*": 0.01, ".*_knee_joint": 0.01, "torso_joint": 0.01},
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit_sim=50, stiffness=0.0, damping=0.0, armature=0.01,
            ),
            # 팔: position 기반 유지
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",   ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                    ".*_five_joint", ".*_three_joint", ".*_six_joint",
                    ".*_four_joint", ".*_zero_joint",  ".*_one_joint", ".*_two_joint",
                ],
                effort_limit_sim=300, stiffness=40.0, damping=10.0,
            ),
        },
    )


# ──────────────────────────────────────────────────────────────
# EnvCfg
# ──────────────────────────────────────────────────────────────

@configclass
class G1IsaacLabEnvCfg(DirectRLEnvCfg):
    episode_length_s: float = 20.0
    decimation: int = 1
    action_space: int = 1   # dummy (DirectRLEnv 필수)
    observation_space: int = 1
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=0.002, render_interval=8)

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    robot: ArticulationCfg = _make_g1_articulation_cfg().replace(
        prim_path="/World/envs/env_.*/Robot"
    )


# ──────────────────────────────────────────────────────────────
# Environment  (상태 제공 + 토크 적용만 담당)
# ──────────────────────────────────────────────────────────────

class G1IsaacLabEnv(DirectRLEnv):
    """
    Isaac Lab G1 환경.

    컨트롤 로직 없음 — run_issac_lab.py (main.cpp 역할) 에서 호출.

    주요 인터페이스:
        get_state()   → qpos, qvel, com, foot positions 등 dict 반환
        set_torque()  → 다리/허리 토크 적용
    """

    cfg: G1IsaacLabEnvCfg

    def __init__(self, cfg: G1IsaacLabEnvCfg, walking_cfg: WalkingConfig,
                 render_mode: str | None = None, **kwargs):
        self.walking_cfg = walking_cfg
        super().__init__(cfg, render_mode, **kwargs)
        self._resolve_indices()
        self._tau_cmd = torch.zeros(self.num_envs, len(self._ctrl_ids), device=self.device)

    # ── 인덱스 매핑 ───────────────────────────────────────────

    def _resolve_indices(self):
        leg_names = [
            "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
            "left_knee_joint",       "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint",      "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]
        waist_names = ["torso_joint"]
        arm_names = [
            "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",    "left_elbow_pitch_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",   "right_elbow_pitch_joint",
        ]

        self._leg_ids,   _ = self.robot.find_joints(leg_names)
        self._waist_ids, _ = self.robot.find_joints(waist_names)
        self._arm_ids,   _ = self.robot.find_joints(arm_names)
        self._ctrl_ids     = list(self._leg_ids) + list(self._waist_ids)  # 다리 12 + 허리 1

        lf_ids,    _ = self.robot.find_bodies("left_ankle_roll_link")
        rf_ids,    _ = self.robot.find_bodies("right_ankle_roll_link")
        torso_ids, _ = self.robot.find_bodies("torso_link")
        self._lf_body    = lf_ids[0]
        self._rf_body    = rf_ids[0]
        self._torso_body = torso_ids[0]

        # floating base nv = 6 + num_joints
        self._is_fb = not self.robot.is_fixed_base
        self._nv    = 6 + self.robot.num_joints if self._is_fb else self.robot.num_joints

    # ── DirectRLEnv 필수 오버라이드 ───────────────────────────

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        pass  # 컨트롤은 run_issac_lab.py에서 set_torque()로 처리

    def _apply_action(self):
        # 다리/허리: effort
        self.robot.set_joint_effort_target(self._tau_cmd, joint_ids=self._ctrl_ids)
        # 팔: 초기 자세 유지 (position)
        arm_pos = self.robot.data.default_joint_pos[:, list(self._arm_ids)]
        self.robot.set_joint_position_target(arm_pos, joint_ids=list(self._arm_ids))

    def _get_observations(self) -> dict:
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        torso_h = (self.robot.data.body_pos_w[:, self._torso_body, 2]
                   - self.scene.env_origins[:, 2])
        terminated = torso_h < 0.3
        time_out   = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        jp         = self.robot.data.default_joint_pos[env_ids]
        jv         = self.robot.data.default_joint_vel[env_ids]
        root_state = self.robot.data.default_root_state[env_ids]
        root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(jp, jv, None, env_ids)

    # ── 외부 인터페이스 ───────────────────────────────────────

    def set_torque(self, tau: torch.Tensor):
        """컨트롤러에서 계산한 토크 (N, n_ctrl) 저장."""
        self._tau_cmd = tau

    def get_state(self) -> dict[str, torch.Tensor]:
        """
        MuJoCo의 qpos/qvel에 대응하는 상태 반환.

        Returns:
            qpos        (N, 7 + num_joints)  root pos(3)+quat(4) + joint_pos
            qvel        (N, 6 + num_joints)  root linvel(3)+angvel(3) + joint_vel
            com_pos     (N, 3)  질량 가중 CoM 위치 (world)
            com_vel     (N, 3)  질량 가중 CoM 속도 (world)
            lf_pos      (N, 3)  왼발 위치 (world)
            rf_pos      (N, 3)  오른발 위치 (world)
            torso_pos   (N, 3)  torso 위치 (world)
            torso_quat  (N, 4)  torso 자세 [w,x,y,z]
            root_linvel (N, 3)
            root_angvel (N, 3)
            joint_pos   (N, num_joints)
            joint_vel   (N, num_joints)
        """
        d       = self.robot.data
        origins = self.scene.env_origins

        # qpos / qvel — MuJoCo 스타일
        qpos = torch.cat([d.root_pos_w - origins, d.root_quat_w, d.joint_pos], dim=-1)
        qvel = torch.cat([d.root_lin_vel_w, d.root_ang_vel_w, d.joint_vel], dim=-1)

        # CoM — 질량 가중 평균 (MuJoCo subtree_com 대응)
        mass       = d.default_mass                                    # (N, B)
        total_mass = mass.sum(-1, keepdim=True)                        # (N, 1)
        com_pos    = (mass.unsqueeze(-1) * d.body_com_pos_w).sum(1) / total_mass
        com_vel    = (mass.unsqueeze(-1) * d.body_com_lin_vel_w).sum(1) / total_mass

        return {
            "qpos":       qpos,
            "qvel":       qvel,
            "com_pos":    com_pos,
            "com_vel":    com_vel,
            "lf_pos":     d.body_pos_w[:, self._lf_body, :],
            "rf_pos":     d.body_pos_w[:, self._rf_body, :],
            "torso_pos":  d.body_pos_w[:, self._torso_body, :],
            "torso_quat": d.body_quat_w[:, self._torso_body, :],
            "root_linvel": d.root_lin_vel_w,
            "root_angvel": d.root_ang_vel_w,
            "joint_pos":  d.joint_pos,
            "joint_vel":  d.joint_vel,
        }

    # Jacobian / 동역학 행렬 (WBC에서 필요 시)
    def get_jacobians(self)    -> torch.Tensor:
        return self.robot.root_physx_view.get_jacobians()

    def get_mass_matrix(self)  -> torch.Tensor:
        return self.robot.root_physx_view.get_generalized_mass_matrices()

    def get_gravity_comp(self) -> torch.Tensor:
        return self.robot.root_physx_view.get_gravity_compensation_forces()
