"""
tau_ff만 적용했을 때 로봇이 안정적으로 서 있는지 테스트.
tau_fb=0으로 고정.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from pytorch.config.g1_config import WalkingConfig
from pytorch.sim.mj_lab_env import G1MjLabEnv
from pytorch.whole_body_controller.DBFC import WBC

def main():
    cfg = WalkingConfig(batch_size=1, device=torch.device('cuda:0'), dtype=torch.float32)
    env = G1MjLabEnv(walking_cfg=cfg, num_envs=1, device='cuda:0')
    env.reset()
    env.forward()

    state = env.get_state()
    cfg.robot_mass = env.total_mass
    cfg.com_height = state['com_pos'][0, 2].item()

    wbc = WBC(cfg, nv=env.nv, na=env.num_joints)

    init_com = state['com_pos'].clone()
    init_rf  = state['rf_pos'].clone()
    init_lf  = state['lf_pos'].clone()

    print(f"init_com = {init_com[0].tolist()}")
    print(f"init_rf  = {init_rf[0].tolist()}")
    print(f"init_lf  = {init_lf[0].tolist()}")
    print(f"robot_mass = {cfg.robot_mass:.4f}, com_height = {cfg.com_height:.4f}")
    print()

    for step in range(500):
        state = env.get_state()
        jacobians  = env.get_jacobians()
        mass_matrix = env.get_mass_matrix()
        nle        = env.get_nle()

        com_pos = state['com_pos']
        com_vel = state['com_vel']
        rf_pos  = state['rf_pos']
        lf_pos  = state['lf_pos']

        # tau_ff만 계산 (ForceOpt + TorqueGen)
        B = 1
        device = cfg.device
        dtype  = cfg.dtype
        m = cfg.robot_mass
        g = cfg.gravity

        from pytorch.whole_body_controller.tasks.balance_task import BalanceTask
        from pytorch.whole_body_controller.Force_Optimizier import ForceOptimizer
        from pytorch.whole_body_controller.whole_body_torque import WholeBodyTorqueGenerator
        from pytorch.constraints.friction_cone import BatchedFrictionCone
        from pytorch.constraints.cop_limits import BatchedCoPLimits

        bal  = wbc.balance_task
        fopt = wbc.force_opt
        tgen = wbc.torque_gen
        fc   = wbc.friction_cone
        cop  = wbc.cop_limits

        com_dot_des = torch.zeros_like(com_pos)
        ddc_pd = bal.update(com_pos, com_vel, init_com, com_dot_des)
        ddc_pd[:, 2] = 0.0

        K = torch.zeros(B, 6, 12, device=device, dtype=dtype)
        K[:, 0, 0]=1; K[:, 1, 1]=1; K[:, 2, 2]=1
        K[:, 0, 6]=1; K[:, 1, 7]=1; K[:, 2, 8]=1

        u_vec = torch.zeros(B, 6, device=device, dtype=dtype)
        u_vec[:, 0] = m * ddc_pd[:, 0]
        u_vec[:, 1] = m * ddc_pd[:, 1]
        u_vec[:, 2] = m * g

        A_fric_r, l_fric_r, u_fric_r = fc.update()
        A_fric_l, l_fric_l, u_fric_l = fc.update()
        A_fric = torch.zeros(B, 10, 12, device=device, dtype=dtype)
        A_fric[:, :5, :3] = A_fric_r
        A_fric[:, 5:, 6:9] = A_fric_l
        l_fric = torch.cat([l_fric_r, l_fric_l], dim=-1)
        u_fric = torch.cat([u_fric_r, u_fric_l], dim=-1)

        contact = torch.ones(B, 2, device=device, dtype=dtype)
        A_cop, l_cop, u_cop = cop.update(contact)

        A_ineq = torch.cat([A_fric, A_cop], dim=1)
        l_ineq = torch.cat([l_fric, l_cop], dim=1)
        u_ineq = torch.cat([u_fric, u_cop], dim=1)

        # 스윙 없음 (양발 접촉)
        import math
        A_sw = torch.zeros(B, 6, 12, device=device, dtype=dtype)
        l_sw = torch.full((B, 6), -math.inf, device=device, dtype=dtype)
        u_sw = torch.full((B, 6), math.inf, device=device, dtype=dtype)
        for j in range(6):
            A_sw[:, j, j] = 1.0
        A_ineq = torch.cat([A_ineq, A_sw], dim=1)
        l_ineq = torch.cat([l_ineq, l_sw], dim=1)
        u_ineq = torch.cat([u_ineq, u_sw], dim=1)

        F_hat = fopt.solve(K, u_vec, A_ineq, l_ineq, u_ineq)
        tau_ff = tgen.compute(mass_matrix, nle, jacobians,
                              env.rf_body_global, env.lf_body_global, F_hat)

        # tau_fb = 0 (피드백 없음)
        tau = tau_ff  # (B, 29)

        leg_tau = tau[:, :len(env.leg_joint_ids)].clamp(-300, 300)
        env.apply_torque(leg_tau)
        env.step()

        if step % 100 == 0:
            com = state['com_pos'][0]
            print(f"step={step:4d}  CoM=({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:.4f})"
                  f"  |tau_ff|={tau_ff[0].norm():.2f}  Fz={F_hat[0,2]:.1f}+{F_hat[0,8]:.1f}")

    print("\n✓ 완료")

if __name__ == '__main__':
    main()
