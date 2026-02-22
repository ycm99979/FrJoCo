"""
WholeBodyTorqueGenerator 디버그 테스트.

시나리오: 초기 자세에서 F_hat (중력 보상) → tau_ff 계산.
C++ standing 시 |tau_ff| ≈ 24, pytorch에서 382 나오던 문제 디버그.

핵심 검증:
  1. J_c 자코비안 행 순서 (MuJoCo [ang,lin] → Pinocchio [lin,ang])
  2. J_c^T @ F_hat 값
  3. G 행렬 조건수
  4. 최종 tau_ff 크기

실행: python -m pytorch.tests.test_torque_generator
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import mujoco

from pytorch.config.g1_config import WalkingConfig
from pytorch.whole_body_controller.whole_body_torque import WholeBodyTorqueGenerator


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
    na = nv - 6  # actuated joints

    print(f"nq={nq}, nv={nv}, na={na}, nbody={nbody}")

    # ── 초기 자세 설정 (C++ knees_bent) ──
    mj_data.qpos[2] = 0.755  # z height
    # quaternion wxyz
    mj_data.qpos[3] = 1.0; mj_data.qpos[4] = 0.0
    mj_data.qpos[5] = 0.0; mj_data.qpos[6] = 0.0

    # 관절 이름 → qpos 인덱스 매핑
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
    print(f"rf_body={rf_id}, lf_body={lf_id}")

    # ── 물리 데이터 계산 ──
    # Mass matrix
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(mj_model, M, mj_data.qM)
    mass_matrix = torch.from_numpy(M).to(device=device, dtype=dtype).unsqueeze(0)

    # NLE (coriolis + gravity)
    nle = torch.from_numpy(mj_data.qfrc_bias.copy()).to(device=device, dtype=dtype).unsqueeze(0)

    # Jacobians (모든 바디)
    jac_all = np.zeros((nbody, 6, nv), dtype=np.float64)
    for b in range(nbody):
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacBody(mj_model, mj_data, jacp, jacr, b)
        jac_all[b, 0:3, :] = jacr  # angular
        jac_all[b, 3:6, :] = jacp  # linear
    jacobians = torch.from_numpy(jac_all).to(device=device, dtype=dtype).unsqueeze(0)

    # ── F_hat: 중력 보상 (양발 대칭) ──
    total_mass = float(sum(mj_model.body_mass))
    fz_half = total_mass * 9.81 / 2.0
    F_hat = torch.zeros(1, 12, device=device, dtype=dtype)
    F_hat[0, 2] = fz_half   # Fz_R
    F_hat[0, 8] = fz_half   # Fz_L
    print(f"\ntotal_mass={total_mass:.4f}, fz_half={fz_half:.4f}")
    print(f"F_hat: {F_hat[0].tolist()}")

    # ── TorqueGenerator ──
    tg = WholeBodyTorqueGenerator(cfg, nv=nv, na=na, nc=12)

    # 자코비안 행 순서 디버그
    J_rf_raw = jacobians[0, rf_id, :, :]  # (6, nv)
    J_lf_raw = jacobians[0, lf_id, :, :]
    print(f"\n--- 자코비안 디버그 (MuJoCo raw: [ang, lin]) ---")
    print(f"J_rf angular (rows 0:3) norm: {J_rf_raw[0:3].norm():.4f}")
    print(f"J_rf linear  (rows 3:6) norm: {J_rf_raw[3:6].norm():.4f}")
    print(f"J_rf linear  row0 (dx/dq): {J_rf_raw[3, :6].tolist()}")
    print(f"J_rf linear  row2 (dz/dq): {J_rf_raw[5, :6].tolist()}")

    # 수동으로 J_c 구성 (Pinocchio 순서: [lin, ang])
    J_c = torch.zeros(1, 12, nv, device=device, dtype=dtype)
    J_c[0, 0:3, :] = J_rf_raw[3:6, :]  # RF linear
    J_c[0, 3:6, :] = J_rf_raw[0:3, :]  # RF angular
    J_c[0, 6:9, :] = J_lf_raw[3:6, :]  # LF linear
    J_c[0, 9:12, :] = J_lf_raw[0:3, :]  # LF angular

    # J_c^T @ F_hat
    Jc_T_F = (J_c[0].T @ F_hat[0]).cpu().numpy()
    print(f"\n--- J_c^T @ F_hat (nv={nv}) ---")
    print(f"  floating base (0:6): {Jc_T_F[:6]}")
    print(f"  joints (6:12):       {Jc_T_F[6:12]}")
    print(f"  |J_c^T @ F_hat|:    {np.linalg.norm(Jc_T_F):.4f}")

    # NLE 확인
    nle_np = nle[0].cpu().numpy()
    print(f"\n--- NLE (qfrc_bias) ---")
    print(f"  floating base (0:6): {nle_np[:6]}")
    print(f"  joints (6:12):       {nle_np[6:12]}")
    print(f"  |NLE|:               {np.linalg.norm(nle_np):.4f}")

    # f[:nv] = -NLE + J_c^T @ F_hat
    f_top = -nle_np + Jc_T_F
    print(f"\n--- f[:nv] = -NLE + J_c^T @ F_hat ---")
    print(f"  floating base (0:6): {f_top[:6]}")
    print(f"  joints (6:12):       {f_top[6:12]}")
    print(f"  |f_top|:             {np.linalg.norm(f_top):.4f}")

    # ── TorqueGenerator 실행 ──
    tau_ff = tg.compute(
        mass_matrix=mass_matrix,
        nle=nle,
        jacobians=jacobians,
        rf_body_idx=rf_id,
        lf_body_idx=lf_id,
        F_hat=F_hat,
    )

    tau_np = tau_ff[0].cpu().numpy()
    print(f"\n{'='*60}")
    print(f"tau_ff (na={na}):")
    print(f"  |tau_ff|:  {np.linalg.norm(tau_np):.4f}")
    print(f"  max:       {np.max(np.abs(tau_np)):.4f}")
    print(f"  first 12:  {tau_np[:12]}")
    print(f"  C++ 기대값: |tau_ff| ≈ 24, max ≈ 15")
    print(f"{'='*60}")

    # 비교: 행 순서 안 바꾸면?
    J_c_wrong = torch.zeros(1, 12, nv, device=device, dtype=dtype)
    J_c_wrong[0, 0:6, :] = J_rf_raw  # [ang, lin] 그대로
    J_c_wrong[0, 6:12, :] = J_lf_raw
    Jc_T_F_wrong = (J_c_wrong[0].T @ F_hat[0]).cpu().numpy()
    f_top_wrong = -nle_np + Jc_T_F_wrong
    print(f"\n--- 비교: 행 순서 안 바꾸면 (MuJoCo raw [ang,lin] 그대로) ---")
    print(f"  |J_c^T @ F_hat| (wrong): {np.linalg.norm(Jc_T_F_wrong):.4f}")
    print(f"  |f_top| (wrong):         {np.linalg.norm(f_top_wrong):.4f}")


if __name__ == "__main__":
    main()
