"""
MuJoCo vs Pinocchio 자코비안 비교 테스트.

MuJoCo mj_jacBody: body frame origin 기준 자코비안
Pinocchio getFrameJacobian(LOCAL_WORLD_ALIGNED): frame origin 기준

핵심 검증:
  1. mj_jacBody vs mj_jacBodyCom 차이 (body frame origin vs body CoM)
  2. 발 링크의 xpos (frame origin) vs xipos (body CoM) 차이
  3. 자코비안 행 순서 확인

실행: python -m pytorch.tests.test_jacobian_compare
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import mujoco


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    xml_path = os.path.join(project_root, "model", "g1", "g1_29dof.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    nv = mj_model.nv
    nq = mj_model.nq

    # 초기 자세
    mj_data.qpos[2] = 0.755
    mj_data.qpos[3] = 1.0
    joint_init = {
        "left_hip_pitch_joint": -0.312,
        "left_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363,
        "right_hip_pitch_joint": -0.312,
        "right_knee_joint": 0.669,
        "right_ankle_pitch_joint": -0.363,
        "waist_pitch_joint": 0.073,
    }
    for name, val in joint_init.items():
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = mj_model.jnt_qposadr[jid]
        mj_data.qpos[qadr] = val

    mujoco.mj_forward(mj_model, mj_data)

    # 바디 인덱스
    rf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    lf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    pelvis_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    print(f"nq={nq}, nv={nv}")
    print(f"rf_body={rf_id}, lf_body={lf_id}, pelvis={pelvis_id}")

    # ── 1. xpos vs xipos (frame origin vs body CoM) ──
    print(f"\n{'='*60}")
    print("1. Body frame origin (xpos) vs Body CoM (xipos)")
    print(f"{'='*60}")

    for name, bid in [("right_ankle_roll", rf_id), ("left_ankle_roll", lf_id), ("pelvis", pelvis_id)]:
        xpos = mj_data.xpos[bid]
        xipos = mj_data.xipos[bid]
        diff = xpos - xipos
        print(f"\n  {name} (body {bid}):")
        print(f"    xpos  (frame origin): {xpos}")
        print(f"    xipos (body CoM):     {xipos}")
        print(f"    diff:                 {diff}")
        print(f"    |diff|:               {np.linalg.norm(diff):.6f}")

    # ── 2. mj_jacBody vs mj_jacBodyCom ──
    print(f"\n{'='*60}")
    print("2. mj_jacBody (frame origin) vs mj_jacBodyCom (body CoM)")
    print(f"{'='*60}")

    for name, bid in [("right_ankle_roll", rf_id), ("left_ankle_roll", lf_id)]:
        # mj_jacBody: body frame origin
        jacp_body = np.zeros((3, nv))
        jacr_body = np.zeros((3, nv))
        mujoco.mj_jacBody(mj_model, mj_data, jacp_body, jacr_body, bid)

        # mj_jacBodyCom: body CoM
        jacp_com = np.zeros((3, nv))
        jacr_com = np.zeros((3, nv))
        mujoco.mj_jacBodyCom(mj_model, mj_data, jacp_com, jacr_com, bid)

        diff_p = jacp_body - jacp_com
        diff_r = jacr_body - jacr_com

        print(f"\n  {name} (body {bid}):")
        print(f"    |jacp_body|:    {np.linalg.norm(jacp_body):.6f}")
        print(f"    |jacp_com|:     {np.linalg.norm(jacp_com):.6f}")
        print(f"    |jacp diff|:    {np.linalg.norm(diff_p):.6f}")
        print(f"    |jacr_body|:    {np.linalg.norm(jacr_body):.6f}")
        print(f"    |jacr_com|:     {np.linalg.norm(jacr_com):.6f}")
        print(f"    |jacr diff|:    {np.linalg.norm(diff_r):.6f}")

        # 선형 자코비안 첫 6열 비교 (floating base)
        print(f"    jacp_body[:, :6]:")
        for r in range(3):
            print(f"      [{jacp_body[r, :6]}]")
        print(f"    jacp_com[:, :6]:")
        for r in range(3):
            print(f"      [{jacp_com[r, :6]}]")

    # ── 3. 사이트 기반 자코비안 (있으면) ──
    print(f"\n{'='*60}")
    print("3. 사이트 목록 (발 관련)")
    print(f"{'='*60}")
    for i in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
        if name and ('foot' in name.lower() or 'ankle' in name.lower()):
            print(f"  site {i}: {name}, pos={mj_data.site_xpos[i]}")

    # ── 4. 관절 이름/순서 확인 ──
    print(f"\n{'='*60}")
    print("4. 관절 순서 (MuJoCo)")
    print(f"{'='*60}")
    for i in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = mj_model.jnt_type[i]
        qadr = mj_model.jnt_qposadr[i]
        vadr = mj_model.jnt_dofadr[i]
        print(f"  joint {i:2d}: {name:35s} type={jtype} qadr={qadr:2d} vadr={vadr:2d}")


if __name__ == "__main__":
    main()
