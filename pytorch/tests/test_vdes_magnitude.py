"""
IK v_des 크기 확인 — C++과 비교
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from pytorch.config.g1_config import WalkingConfig
from pytorch.sim.mj_lab_env import G1MjLabEnv
from pytorch.whole_body_controller.whole_body_ik import WholeBodyIK

def main():
    cfg = WalkingConfig(batch_size=1, device=torch.device('cuda:0'), dtype=torch.float32)
    env = G1MjLabEnv(walking_cfg=cfg, num_envs=1, device='cuda:0')
    env.reset()
    env.forward()

    state = env.get_state()
    cfg.robot_mass = env.total_mass
    cfg.com_height = state['com_pos'][0, 2].item()

    ik = WholeBodyIK(cfg, nv=env.nv, na=env.num_joints)

    init_com = state['com_pos'].clone()
    init_rf  = state['rf_pos'].clone()
    init_lf  = state['lf_pos'].clone()
    init_rf_ori = state['rf_ori'].clone()
    init_lf_ori = state['lf_ori'].clone()

    jacobians = env.get_jacobians()
    com_jac   = env.get_com_jacobian()
    q_full    = state['q_full']
    dq_full   = state['dq_full']

    print("=== Step 1: 오차 없는 상태 ===")
    ik.compute(
        jacobians=jacobians, q_curr=q_full, dq_curr=dq_full,
        com_jac=com_jac, com_pos=init_com,
        rf_body_idx=env.rf_body_global, lf_body_idx=env.lf_body_global,
        com_des=init_com, rf_pos_des=init_rf, lf_pos_des=init_lf,
        rf_pos_curr=init_rf, lf_pos_curr=init_lf,
        rf_ori_curr=init_rf_ori, lf_ori_curr=init_lf_ori,
        rf_ori_des=init_rf_ori, lf_ori_des=init_lf_ori,
    )
    print(f"  |v_des|       = {ik.v_des[0].norm():.6f}")
    print(f"  |v_des[6:]|   = {ik.v_des[0, 6:].norm():.6f}")
    print(f"  |v_des[:6]|   = {ik.v_des[0, :6].norm():.6f}")
    print(f"  v_des[:6]     = {ik.v_des[0, :6].tolist()}")
    print(f"  v_des[6:12]   = {ik.v_des[0, 6:12].tolist()}")

    print("\n=== Step 2: CoM 오차 1cm ===")
    com_des_shifted = init_com.clone()
    com_des_shifted[:, 0] += 0.01  # 1cm 앞으로
    ik2 = WholeBodyIK(cfg, nv=env.nv, na=env.num_joints)
    ik2.compute(
        jacobians=jacobians, q_curr=q_full, dq_curr=dq_full,
        com_jac=com_jac, com_pos=init_com,
        rf_body_idx=env.rf_body_global, lf_body_idx=env.lf_body_global,
        com_des=com_des_shifted, rf_pos_des=init_rf, lf_pos_des=init_lf,
        rf_pos_curr=init_rf, lf_pos_curr=init_lf,
        rf_ori_curr=init_rf_ori, lf_ori_curr=init_lf_ori,
        rf_ori_des=init_rf_ori, lf_ori_des=init_lf_ori,
    )
    print(f"  |v_des|       = {ik2.v_des[0].norm():.4f}")
    print(f"  |v_des[6:]|   = {ik2.v_des[0, 6:].norm():.4f}")
    tau_fb = ik2.Kp * (ik2.q_des[0, 7:] - q_full[0, 7:]) + ik2.Kd * ik2.v_des[0, 6:]
    print(f"  |tau_fb|      = {tau_fb.norm():.4f}")
    print(f"  v_des[6:12]   = {ik2.v_des[0, 6:12].tolist()}")

    print("\n=== Step 3: 발 위치 오차 1cm ===")
    rf_des_shifted = init_rf.clone()
    rf_des_shifted[:, 0] += 0.01
    ik3 = WholeBodyIK(cfg, nv=env.nv, na=env.num_joints)
    ik3.compute(
        jacobians=jacobians, q_curr=q_full, dq_curr=dq_full,
        com_jac=com_jac, com_pos=init_com,
        rf_body_idx=env.rf_body_global, lf_body_idx=env.lf_body_global,
        com_des=init_com, rf_pos_des=rf_des_shifted, lf_pos_des=init_lf,
        rf_pos_curr=init_rf, lf_pos_curr=init_lf,
        rf_ori_curr=init_rf_ori, lf_ori_curr=init_lf_ori,
        rf_ori_des=init_rf_ori, lf_ori_des=init_lf_ori,
    )
    print(f"  |v_des|       = {ik3.v_des[0].norm():.4f}")
    print(f"  |v_des[6:]|   = {ik3.v_des[0, 6:].norm():.4f}")
    tau_fb3 = ik3.Kp * (ik3.q_des[0, 7:] - q_full[0, 7:]) + ik3.Kd * ik3.v_des[0, 6:]
    print(f"  |tau_fb|      = {tau_fb3.norm():.4f}")

    print("\n✓ 완료")

if __name__ == '__main__':
    main()
