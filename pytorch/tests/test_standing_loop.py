"""
Standing Loop 통합 디버그 테스트.

시나리오: MuJoCo 초기 자세에서 standing_loop 1스텝 실행.
tau_ff + tau_fb 분리 출력, C++ 값과 비교.

C++ standing 기대값:
  |tau_ff| ≈ 24, |tau_fb| ≈ 0 (첫 스텝)
  F_hat ≈ [0, 0, ~172, 0, ~-6, 0, 0, 0, ~172, 0, ~-6, 0]

실행: python -m pytorch.tests.test_standing_loop
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import mujoco

from pytorch.config.g1_config import WalkingConfig
from pytorch.whole_body_controller.DBFC import WBC


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    cfg = WalkingConfig(batch_size=1, device=device, dtype=dtype)

    # ── MuJoCo 모델 로드 ──
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    xml_path = os.path.join(project_root, "model", "g1", "g1_29dof.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    nv = mj_model.nv
    nq = mj_model.nq
    nbody = mj_model.nbody
    na = nv - 6

    # 시뮬레이터에서 물리 파라미터 추출
    cfg.robot_mass = float(sum(mj_model.body_mass))

    print(f"nq={nq}, nv={nv}, na={na}, nbody={nbody}")
    print(f"robot_mass={cfg.robot_mass:.4f}")
    print(f"IK_KP={cfg.ik_kp}, IK_KD={cfg.ik_kd}")
    print(f"BAL_KP={cfg.bal_kp}, BAL_KD={cfg.bal_kd}")

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
    q_mj = torch.from_numpy(mj_data.qpos[:nq].copy()).to(device=device, dtype=dtype).unsqueeze(0)
    q_full = q_mj.clone()
    q_full[0, 3] = q_mj[0, 4]  # x
    q_full[0, 4] = q_mj[0, 5]  # y
    q_full[0, 5] = q_mj[0, 6]  # z
    q_full[0, 6] = q_mj[0, 3]  # w
    dq_full = torch.zeros(1, nv, device=device, dtype=dtype)

    com_pos = torch.from_numpy(mj_data.subtree_com[1].copy()).to(device=device, dtype=dtype).unsqueeze(0)
    com_vel = torch.zeros(1, 3, device=device, dtype=dtype)
    cfg.com_height = com_pos[0, 2].item()

    rf_pos = torch.from_numpy(mj_data.xpos[rf_id].copy()).to(device=device, dtype=dtype).unsqueeze(0)
    lf_pos = torch.from_numpy(mj_data.xpos[lf_id].copy()).to(device=device, dtype=dtype).unsqueeze(0)
    rf_mat = torch.from_numpy(mj_data.xmat[rf_id].reshape(3, 3).copy()).to(device=device, dtype=dtype).unsqueeze(0)
    lf_mat = torch.from_numpy(mj_data.xmat[lf_id].reshape(3, 3).copy()).to(device=device, dtype=dtype).unsqueeze(0)

    print(f"com_pos: {com_pos[0].tolist()}")
    print(f"com_height: {cfg.com_height:.6f}")
    print(f"rf_pos: {rf_pos[0].tolist()}")
    print(f"lf_pos: {lf_pos[0].tolist()}")

    # Mass matrix
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(mj_model, M, mj_data.qM)
    mass_matrix = torch.from_numpy(M).to(device=device, dtype=dtype).unsqueeze(0)

    # NLE
    nle = torch.from_numpy(mj_data.qfrc_bias.copy()).to(device=device, dtype=dtype).unsqueeze(0)

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

    # ── WBC 초기화 ──
    wbc = WBC(cfg, nv=nv, na=na, nc=12)

    # ── Standing Loop: com_des = com_curr ──
    com_des = com_pos.clone()
    contact = torch.ones(1, 2, device=device, dtype=dtype)

    print(f"\n{'='*60}")
    print("Standing Loop Step 1")
    print(f"{'='*60}")

    tau = wbc.compute_ik(
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
        rf_ori_des=rf_mat,   # 현재 자세 유지
        lf_ori_des=lf_mat,
        mass_matrix=mass_matrix,
        nle=nle,
        com_vel=com_vel,
        contact_state=contact,
    )

    tau_np = tau[0].cpu().numpy()
    print(f"\n  |tau_total|: {np.linalg.norm(tau_np):.4f}")
    print(f"  max|tau|:    {np.max(np.abs(tau_np)):.4f}")
    print(f"  tau[:6]:     {tau_np[:6]}")
    print(f"  tau[6:12]:   {tau_np[6:12]}")

    # tau_ff, tau_fb 분리 (WBC 내부에서 이미 출력됨)
    # F_hat 확인
    F_hat = wbc.force_opt.opt_F[0].cpu().numpy()
    print(f"\n  F_hat: {F_hat}")
    print(f"  |F_hat|: {np.linalg.norm(F_hat):.4f}")
    print(f"  Fz_R={F_hat[2]:.4f}, Fz_L={F_hat[8]:.4f}")
    print(f"  Fz_total={F_hat[2]+F_hat[8]:.4f} (mg={cfg.robot_mass*9.81:.4f})")

    print(f"\n  C++ 기대값:")
    print(f"    |tau_ff| ≈ 24, |tau_fb| ≈ 0")
    print(f"    |tau_total| ≈ 24")
    print(f"    F_hat ≈ [0, 0, 172, 0, -6, 0, 0, 0, 172, 0, -6, 0]")

    # ── Step 2: 같은 상태로 한번 더 (q_des가 업데이트된 상태) ──
    print(f"\n{'='*60}")
    print("Standing Loop Step 2 (same state, q_des updated)")
    print(f"{'='*60}")

    tau2 = wbc.compute_ik(
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
        mass_matrix=mass_matrix,
        nle=nle,
        com_vel=com_vel,
        contact_state=contact,
    )

    tau2_np = tau2[0].cpu().numpy()
    print(f"\n  |tau_total|: {np.linalg.norm(tau2_np):.4f}")
    print(f"  max|tau|:    {np.max(np.abs(tau2_np)):.4f}")


if __name__ == "__main__":
    main()
