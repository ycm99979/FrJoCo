"""
초기 자세 적용 여부 디버그 테스트.

문제: env.reset() 후 qpos[7:] = 0 (zero pose)
기대: qpos[7:12] = [-0.312, 0, 0, 0.669, -0.363, ...]
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import mujoco
import numpy as np

from pytorch.config.g1_config import WalkingConfig
from pytorch.sim.mj_lab_env import G1MjLabEnv, _INIT_STATE

def main():
    cfg = WalkingConfig(batch_size=1, device=torch.device("cuda:0"), dtype=torch.float32)
    env = G1MjLabEnv(walking_cfg=cfg, num_envs=1, device="cuda:0")

    print("=" * 60)
    print("1. reset() 전 qpos")
    print(f"   sim.data.qpos[0, :13] = {env.sim.data.qpos[0, :13].tolist()}")

    env.reset()
    env.forward()

    print("\n2. reset() + forward() 후 qpos")
    qpos = env.sim.data.qpos[0].detach().cpu().numpy()
    print(f"   qpos[0:7]  (freejoint) = {qpos[:7]}")
    print(f"   qpos[7:13] (L leg)     = {qpos[7:13]}")
    print(f"   qpos[13:19](R leg)     = {qpos[13:19]}")
    print(f"   qpos[19:22](waist)     = {qpos[19:22]}")

    print("\n3. 기대값 (_INIT_STATE)")
    expected = {
        "left_hip_pitch_joint":  -0.312,
        "left_knee_joint":        0.669,
        "left_ankle_pitch_joint": -0.363,
        "right_hip_pitch_joint": -0.312,
        "right_knee_joint":       0.669,
        "right_ankle_pitch_joint":-0.363,
    }
    for k, v in expected.items():
        print(f"   {k}: {v}")

    print("\n4. CPU mj_data 상태 (jacobian 계산용)")
    env._sync_cpu_data(0)
    print(f"   _mj_data.qpos[7:13] = {env._mj_data.qpos[7:13]}")
    print(f"   _mj_data.qpos[13:19]= {env._mj_data.qpos[13:19]}")

    print("\n5. get_state() 결과")
    state = env.get_state()
    print(f"   com_pos  = {state['com_pos'][0].tolist()}")
    print(f"   lf_pos   = {state['lf_pos'][0].tolist()}")
    print(f"   rf_pos   = {state['rf_pos'][0].tolist()}")
    print(f"   q_full[7:13] = {state['q_full'][0, 7:13].tolist()}")

    print("\n6. robot.data.joint_pos (entity 기준)")
    print(f"   joint_pos[0, :12] = {env.robot.data.joint_pos[0, :12].tolist()}")

    print("\n7. mjlab scene entity 초기 상태 확인")
    # _INIT_STATE joint_pos dict 출력
    print(f"   _INIT_STATE.joint_pos = {_INIT_STATE.joint_pos}")

    print("\n8. 수동으로 초기 자세 적용 시도")
    # MuJoCo 관절 이름 → qpos 인덱스 매핑
    mj_model = env._mj_model
    joint_name_to_qpos = {}
    for i in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qadr = mj_model.jnt_qposadr[i]
        joint_name_to_qpos[name] = qadr
        
    print("   관절 이름 → qpos 인덱스:")
    for name, idx in list(joint_name_to_qpos.items())[:15]:
        print(f"     {name}: qpos[{idx}]")

    # 수동 적용
    qpos_manual = env.sim.data.qpos[0].detach().cpu().numpy().copy()
    for jname, jval in _INIT_STATE.joint_pos.items():
        if jname in joint_name_to_qpos:
            idx = joint_name_to_qpos[jname]
            qpos_manual[idx] = jval
    
    print(f"\n   수동 적용 후 qpos[7:13] = {qpos_manual[7:13]}")
    print(f"   수동 적용 후 qpos[13:19]= {qpos_manual[13:19]}")

    print("\n9. GPU qpos에 직접 쓰기 테스트")
    qpos_tensor = torch.from_numpy(qpos_manual).to(device="cuda:0", dtype=torch.float32)
    env.sim.data.qpos[0, :] = qpos_tensor
    env.sim.forward()
    env._sync_cpu_data(0)
    
    state2 = env.get_state()
    print(f"   com_pos after manual = {state2['com_pos'][0].tolist()}")
    print(f"   q_full[7:13]         = {state2['q_full'][0, 7:13].tolist()}")

    print("\n✓ 테스트 완료")

if __name__ == "__main__":
    main()
