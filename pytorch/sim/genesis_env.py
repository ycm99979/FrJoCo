"""G1 Humanoid Genesis 병렬 시뮬레이션 환경."""

import torch
import numpy as np

try:
    import genesis as gs
except ImportError:
    gs = None


class G1GenesisEnv:
    """G1 휴머노이드 Genesis 병렬 환경 - 토크/위치 제어 인터페이스."""

    LEG_JOINT_NAMES = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]
    WAIST_JOINT_NAMES = [
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    ]
    ARM_JOINT_NAMES = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
    ]
    ALL_JOINT_NAMES = LEG_JOINT_NAMES + WAIST_JOINT_NAMES + ARM_JOINT_NAMES

    def __init__(self, num_envs=1, mjcf_path=None, dt=0.002,
                 show_viewer=False, backend="gpu"):
        assert gs is not None, "pip install genesis-world"
        if mjcf_path is None:
            mjcf_path = "xml/humanoid.xml"

        self.num_envs = num_envs
        self.dt = dt

        gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)
        self.device = gs.device

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=2),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -1.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.7),
                camera_fov=40,
                max_FPS=int(1.0 / dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=mjcf_path, pos=(0.0, 0.0, 0.793)),
        )
        self.scene.build(n_envs=num_envs, env_spacing=(2.0, 2.0))

        self.leg_dof_idx = self._get_dof_indices(self.LEG_JOINT_NAMES)
        self.waist_dof_idx = self._get_dof_indices(self.WAIST_JOINT_NAMES)
        self.arm_dof_idx = self._get_dof_indices(self.ARM_JOINT_NAMES)
        self.all_dof_idx = self._get_dof_indices(self.ALL_JOINT_NAMES)
        self.num_dofs = len(self.all_dof_idx)
        self.init_qpos = self.robot.get_qpos().clone()

    def _get_dof_indices(self, joint_names):
        return [self.robot.get_joint(name).dof_start for name in joint_names]

    def step_torque(self, tau):
        self.robot.control_dofs_force(tau, self.all_dof_idx)
        self.scene.step()

    def step_position(self, q_target, dof_idx=None):
        idx = dof_idx if dof_idx is not None else self.all_dof_idx
        self.robot.control_dofs_position(q_target, idx)
        self.scene.step()

    def get_obs(self):
        return {
            'base_pos': self.robot.get_pos(),
            'base_quat': self.robot.get_quat(),
            'base_vel': self.robot.get_vel(),
            'base_ang_vel': self.robot.get_ang(),
            'dof_pos': self.robot.get_dofs_position(self.all_dof_idx),
            'dof_vel': self.robot.get_dofs_velocity(self.all_dof_idx),
        }

    def get_base_height(self):
        return self.robot.get_pos()[:, 2]

    def set_pd_gains(self, kp_legs=200.0, kd_legs=10.0,
                     kp_ankles=50.0, kd_ankles=5.0,
                     kp_arms=100.0, kd_arms=5.0):
        kp = [kp_legs] * len(self.LEG_JOINT_NAMES)
        kd = [kd_legs] * len(self.LEG_JOINT_NAMES)
        for i, name in enumerate(self.LEG_JOINT_NAMES):
            if "ankle" in name:
                kp[i], kd[i] = kp_ankles, kd_ankles
        self.robot.set_dofs_kp(kp, self.leg_dof_idx)
        self.robot.set_dofs_kv(kd, self.leg_dof_idx)
        self.robot.set_dofs_kp([kp_arms] * len(self.ARM_JOINT_NAMES), self.arm_dof_idx)
        self.robot.set_dofs_kv([kd_arms] * len(self.ARM_JOINT_NAMES), self.arm_dof_idx)

    def reset(self, envs_idx=None):
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True)

    def is_fallen(self):
        return self.get_base_height() < 0.3
