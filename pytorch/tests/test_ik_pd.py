"""
WholeBodyIK + PD 토크 디버그 테스트.

시나리오: 초기 자세에서 IK 1스텝 → PD 토크 계산.
C++ standing 시 |tau_fb| ≈ 0~11, pytorch에서 폭발하던 문제 디버그.

핵심 검증:
  1. IK 자코비안 스택 구성 (행 순서)
  2. dx_err (태스크 오차) — 초기 자세에서 거의 0이어야 함
  3. v_des (목표 관절 속도) — 작아야 함
  4. q_des - q_curr (위치 오차) — 작아야 함
  5. PD 토크 크기

실행: python -m pytorch.tests.test_ik_pd
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import mujoco

from pytorch.config.g1_config import WalkingConfig
from pytorch.whole_body_controller.whole_body_ik import WholeBodyIK


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    cfg = WalkingConfig(batch_size=1, device=device, dtype=dtype)

    print(f"IK_KP={cfg.ik_kp}, IK_KD={cfg.ik_kd}")
    print(f"IK_DAMPING={cfg.ik_damping}, IK_V_MAX={cfg.ik_v_max}")

    # ── MuJoCo 모델 로드 ──
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    xml_path = os.path.join(project_root, "model", "g1", "g1_29dof.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    nv = mj_model.nv
    nq = mj_model.nq
    nbody = mj_model.nbody
    na = nv - 6

    print(f"nq={nq}, nv={nv}, na={na}")

    # ── 초기 자세 설정 ──
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

    # ── 바디 인덱스 ──
    rf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    lf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")

    # ── 상태 추출 ──
    # q_full: MuJoCo wxyz → xyzw 변환
    q_mj = torch.from_numpy(mj_data.qpos[:nq].copy()).to(device=device, dtype=dtype).unsqueeze(0)
    q_full = q_mj.clone()
    q_full[0, 3] = q_mj[0, 4]  # x
    q_full[0, 4] = q_mj[0, 5]  # y
    q_full[0, 5] = q_mj[0, 6]  # z
    q_full[0, 6] = q_mj[0, 3]  # w
    dq_full = torch.zeros(1, nv, device=device, dtype=dtype)

    # CoM
    com_pos = torch.from_numpy(mj_data.subtree_com[1].copy()).to(device=device, dtype=dtype).unsqueeze(0)
    com_des = com_pos.clone()  # standing: 현재 위치 유지

    # 발 위치/자세
    rf_pos = torch.from_numpy(mj_data.xpos[rf_id].copy()).to(device=device, dtype=dtype).unsqueeze(0)
    lf_pos = torch.from_numpy(mj_data.xpos[lf_id].copy()).to(device=device, dtype=dtype).unsqueeze(0)
    rf_mat = torch.from_numpy(mj_data.xmat[rf_id].reshape(3, 3).copy()).to(device=device, dtype=dtype).unsqueeze(0)
    lf_mat = torch.from_numpy(mj_data.xmat[lf_id].reshape(3, 3).copy()).to(device=device, dtype=dtype).unsqueeze(0)

    print(f"\ncom_pos: {com_pos[0].tolist()}")
    print(f"rf_pos:  {rf_pos[0].tolist()}")
    print(f"lf_pos:  {lf_pos[0].tolist()}")
    print(f"q_full[3:7] (xyzw): {q_full[0, 3:7].tolist()}")

    # Jacobians
    jac_all = np.zeros((nbody, 6, nv), dtype=np.float64)
    for b in range(nbody):
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacBody(mj_model, mj_data, jacp, jacr, b)
        jac_all[b, 0:3, :] = jacr
        jac_all[b, 3:6, :] = jacp
    jacobians = torch.from_numpy(jac_all).to(device=device, dtype=dtype).unsqueeze(0)

    # CoM jacobian
    total_mass = 0.0
    com_jac_np = np.zeros((3, nv))
    for b in range(1, nbody):
        mb = mj_model.body_mass[b]
        if mb < 1e-10:
            continue
        jacp = np.zeros((3, nv))
        mujoco.mj_jacBody(mj_model, mj_data, jacp, None, b)
        com_jac_np += mb * jacp
        total_mass += mb
    com_jac_np /= total_mass
    com_jac = torch.from_numpy(com_jac_np).to(device=device, dtype=dtype).unsqueeze(0)

    # ── IK 테스트 ──
    ik = WholeBodyIK(cfg, nv=nv, na=na)

    print(f"\n{'='*60}")
    print("IK Step 1: 초기 자세에서 com_des = com_curr, foot_des = foot_curr")
    print(f"{'='*60}")

    ik.compute(
        jacobians=jacobians,
        q_curr=q_full,
        dq_curr=dq_full,
        com_jac=com_jac,
        com_pos=com_pos,
        rf_body_idx=rf_id,
        lf_body_idx=lf_id,
        com_des=com_des,
        rf_pos_des=rf_pos,
        lf_pos_des=lf_pos,
        rf_pos_curr=rf_pos,
        lf_pos_curr=lf_pos,
        rf_ori_curr=rf_mat,
        lf_ori_curr=lf_mat,
    )

    print(f"\n  |v_des|:     {ik.v_des[0].norm():.6f}")
    print(f"  v_des[:6]:   {ik.v_des[0, :6].tolist()}")
    print(f"  v_des[6:12]: {ik.v_des[0, 6:12].tolist()}")

    q_err = ik.q_des[0, 7:] - q_full[0, 7:]
    v_err = ik.v_des[0, 6:] - dq_full[0, 6:]
    print(f"\n  |q_err|:     {q_err.norm():.6f}")
    print(f"  |v_err|:     {v_err.norm():.6f}")

    tau_fb = ik.compute_pd_torque(q_full, dq_full)
    print(f"\n  |tau_fb|:    {tau_fb[0].norm():.4f}")
    print(f"  max|tau_fb|: {tau_fb[0].abs().max():.4f}")
    print(f"  tau_fb[:6]:  {tau_fb[0, :6].tolist()}")
    print(f"  C++ 기대값:  |tau_fb| ≈ 0 (첫 스텝, 오차 없음)")

    # ── IK Step 2: com_des를 살짝 이동 ──
    print(f"\n{'='*60}")
    print("IK Step 2: com_des.x += 0.01 (1cm 전방 이동)")
    print(f"{'='*60}")

    com_des2 = com_pos.clone()
    com_des2[0, 0] += 0.01

    ik.compute(
        jacobians=jacobians,
        q_curr=q_full,
        dq_curr=dq_full,
        com_jac=com_jac,
        com_pos=com_pos,
        rf_body_idx=rf_id,
        lf_body_idx=lf_id,
        com_des=com_des2,
        rf_pos_des=rf_pos,
        lf_pos_des=lf_pos,
        rf_pos_curr=rf_pos,
        lf_pos_curr=lf_pos,
        rf_ori_curr=rf_mat,
        lf_ori_curr=lf_mat,
    )

    print(f"\n  |v_des|:     {ik.v_des[0].norm():.6f}")
    q_err2 = ik.q_des[0, 7:] - q_full[0, 7:]
    v_err2 = ik.v_des[0, 6:] - dq_full[0, 6:]
    print(f"  |q_err|:     {q_err2.norm():.6f}")
    print(f"  |v_err|:     {v_err2.norm():.6f}")

    tau_fb2 = ik.compute_pd_torque(q_full, dq_full)
    print(f"\n  |tau_fb|:    {tau_fb2[0].norm():.4f}")
    print(f"  max|tau_fb|: {tau_fb2[0].abs().max():.4f}")


if __name__ == "__main__":
    main()
