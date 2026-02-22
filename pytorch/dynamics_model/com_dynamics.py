# Dynamic Balance Force Control for Compliant Humanoid Robots
# II. COM Dynamics model의 식 (1) ~ (5)까지 구현

import torch
from pytorch.config.g1_config import WalkingConfig
from pytorch.utils.batch_utils import batched_skew_symmetric


class CenterOfMassDynamics:
    def __init__(self, cfg: WalkingConfig):
        self.num_envs = cfg.batch_size
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.m = cfg.robot_mass
        self.g = cfg.gravity

        # 식 (2) D1 = [I 0 I 0]
        I = torch.eye(3, device=cfg.device, dtype=cfg.dtype)
        D1 = torch.zeros((3, 12), device=cfg.device, dtype=cfg.dtype)
        D1[:, 0:3] = I  # right foot
        D1[:, 6:9] = I  # left foot
        self.D1 = D1.unsqueeze(0).expand(cfg.batch_size, -1, -1)

    def update(self, com_pos, l_foot_pos, r_foot_pos, desired_ddcom, dL):
        r_R = r_foot_pos - com_pos
        r_L = l_foot_pos - com_pos

        r_R_skew = batched_skew_symmetric(r_R)
        r_L_skew = batched_skew_symmetric(r_L)

        I = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0).expand(self.num_envs, -1, -1)

        # 식 (3) D2 = [P_Rx I P_Lx I]
        D2 = torch.zeros((self.num_envs, 3, 12), device=self.device, dtype=self.dtype)
        D2[:, :, 0:3]  = r_R_skew
        D2[:, :, 3:6]  = I
        D2[:, :, 6:9]  = r_L_skew
        D2[:, :, 9:12] = I

        # 행렬 가로로 쌓기 [D1; D2]
        K = torch.cat([self.D1, D2], dim=1)

        # 식 (1)의 우변
        F_g = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        F_g[:, 2] = self.m * self.g

        # 식 (5)
        u_head = self.m * desired_ddcom + F_g
        u = torch.cat([u_head, dL], dim=1)

        return K, u
