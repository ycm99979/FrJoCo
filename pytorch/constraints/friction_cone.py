# Dynamic Balance Force Control for Compliant Humanoid Robots
# II. COM Dynamics model의 식 (11) ~(13) 까지 구현

import torch
from pytorch.config.g1_config import WalkingConfig


class BatchedFrictionCone:

    def __init__(self, cfg: WalkingConfig):
        self.num_envs = cfg.batch_size
        self.device = cfg.device

        mu = cfg.friction_mu
        mu_eff = mu / (2.0 ** 0.5)
        INF = float('inf')

        A_base = torch.zeros((5, 3), device=cfg.device, dtype=cfg.dtype)
        A_base[0, 0], A_base[0, 2] =  1.0, -mu_eff
        A_base[1, 0], A_base[1, 2] = -1.0, -mu_eff
        A_base[2, 1], A_base[2, 2] =  1.0, -mu_eff
        A_base[3, 1], A_base[3, 2] = -1.0, -mu_eff
        A_base[4, 2] = 1.0

        self.A = A_base.unsqueeze(0).expand(cfg.batch_size, -1, -1)

        self.l = torch.tensor([-INF, -INF, -INF, -INF, 0.0], device=cfg.device, dtype=cfg.dtype).expand(cfg.batch_size, -1)
        self.u = torch.tensor([ 0.0,  0.0,  0.0,  0.0, INF], device=cfg.device, dtype=cfg.dtype).expand(cfg.batch_size, -1)

    def update(self, state=None):
        return self.A, self.l, self.u