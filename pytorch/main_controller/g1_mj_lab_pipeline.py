"""
G1 Walking Controller Pipeline — mjlab 버전.

C++ g1_walking_controller 대응, Isaac Lab pipeline과 동일 구조.
mjlab 환경에서 jacobian/mass_matrix를 CPU MuJoCo API로 계산.

구조:
  생성자: footstep plan → ZMP ref → foot trajectory → CoM ref (오프라인)
  mpc_loop(100Hz): 상태 보정 → MPC QP → 상태 업데이트
  wbc_loop(1kHz):  IK → ForceOptimizer → TorqueGenerator → τ
  standing_loop:   MPC 없이 초기 자세 유지

mjlab API 사용:
  - G1MjLabEnv.get_jacobians()   → CPU mujoco → torch
  - G1MjLabEnv.get_mass_matrix() → CPU mujoco → torch
  - G1MjLabEnv.get_nle()         → qfrc_bias
  - G1MjLabEnv.get_com_jacobian()→ 질량 가중 CoM jacobian
"""

from pytorch.main_controller.g1_isaac_lab_pipeline import G1WalkingPipeline


class G1MjLabPipeline(G1WalkingPipeline):
    """mjlab용 G1 보행 파이프라인.

    G1WalkingPipeline을 그대로 상속.
    MPC/WBC/trajectory 로직은 시뮬레이터 독립적이므로 변경 없음.

    mjlab 특화 로직은 run_mjlab.py에서 상태 추출 → 파이프라인 호출 시 처리.
    (jacobian, mass_matrix 등은 G1MjLabEnv가 제공)
    """
    pass
