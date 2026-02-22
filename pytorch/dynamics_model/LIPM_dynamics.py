# Kajita Introduce to humanoid robotics 
# 3D LIPM 모델에서 가져옴

import torch
from pytorch.config.g1_config import WalkingConfig

# 정적 동역학 행렬
class LIPMdynamics:
    def __init__(self, cfg: WalkingConfig):
        self.device = cfg.device
        self.num_envs = cfg.batch_size
        dt = cfg.mpc_dt  # LIPM은 MPC 주기로 이산화 (MPC_DT = 0.01s)
        z_c = cfg.com_height
        gravity = cfg.gravity
        dtype = cfg.dtype

        # 이산화된 A
        A = torch.eye(3, device=cfg.device, dtype=dtype)
        A[0, 1] = dt
        A[0, 2] = 0.5 * (dt ** 2)
        A[1, 2] = dt

        # 이산화된 B
        B = torch.zeros((3, 1), device=cfg.device, dtype=dtype)
        B[0, 0] = (dt ** 3) / 6.0
        B[1, 0] = (dt ** 2) / 2.0
        B[2, 0] = dt

        # 이산화된 C
        C = torch.tensor([[1.0, 0.0, -z_c / gravity]], device=cfg.device, dtype=dtype)

        # (num_envs, 3, 3), (num_envs, 3, 1), (num_envs, 1, 3) 형태로 확장하여 저장
        self.Ad = A.unsqueeze(0).expand(cfg.batch_size, -1, -1)
        self.Bd = B.unsqueeze(0).expand(cfg.batch_size, -1, -1)
        self.Cd = C.unsqueeze(0).expand(cfg.batch_size, -1, -1)

    def step(self, x_current, u):

        # x_current = CoM의 현재 위치 속도 가속도
        # u = CoM의 저크

        # x_next = Ad @ x + Bd @ u
        x_next = torch.bmm(self.Ad, x_current) + torch.bmm(self.Bd, u)

        # zmp = Cd @ x
        zmp = torch.bmm(self.Cd, x_next)

        return x_next, zmp
