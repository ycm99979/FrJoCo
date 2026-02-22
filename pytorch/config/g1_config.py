"""G1 보행 제어 설정 — config.hpp 대응."""

from dataclasses import dataclass, field
import torch


@dataclass
class WalkingConfig:

    # ── 디바이스 / 배치 ──────────────────────────────────────
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: torch.dtype   = torch.float32
    batch_size: int      = 1

    # ── 제어 주기 (config.hpp: MJ_TIMESTEP, WBC_DT, MPC_DT) ─
    dt: float     = 0.001   # WBC_DT  = 1/1000
    mpc_dt: float = 0.01    # MPC_DT  = 1/100
    @property
    def mpc_decimation(self) -> int:
        return round(self.mpc_dt / self.dt)   # MPC_DECIMATION = 10

    # ── 물리 (config.hpp: GRAVITY) ───────────────────────────
    gravity: float = 9.81

    # ── 로봇 (config.hpp: COM_HEIGHT, 질량은 Pinocchio에서 추출) ─
    robot_mass: float  = 35.0    # 기본값; 시뮬레이터에서 덮어씀
    com_height: float  = 0.69   # COM_HEIGHT

    # ── 보행 파라미터 (config.hpp: STEP_*, DSP_TIME, N_STEPS) ─
    n_steps: int       = 20         # N_STEPS
    step_length: float = 0.1        # STEP_LENGTH
    step_width: float  = 0.1185     # STEP_WIDTH
    step_height: float = 0.06       # STEP_HEIGHT
    step_time: float   = 0.8        # STEP_TIME
    dsp_time: float    = 0.12       # DSP_TIME

    # ── MPC (config.hpp: MPC_HORIZON, MPC_ALPHA, MPC_GAMMA) ──
    mpc_horizon: int   = 160        # MPC_HORIZON
    mpc_alpha: float   = 1e-6       # jerk 페널티
    mpc_gamma: float   = 1.0        # ZMP 추종 페널티

    # ── WBC IK (config.hpp: IK_KP, IK_KD, IK_DAMPING, IK_V_MAX) ─
    ik_kp: float      = 300.0      # IK_KP (C++: 300)
    ik_kd: float      = 35.0       # IK_KD (C++: 35)
    ik_damping: float = 1e-2        # IK_DAMPING
    ik_v_max: float   = 30.0        # IK_V_MAX  (rad/s)

    # ── Balance (config.hpp: BAL_KP, BAL_KD) ─────────────────
    bal_kp: float = 20.0            # BAL_KP
    bal_kd: float = 9.0             # BAL_KD

    # ── Force Optimizer (config.hpp: FORCE_OPT_REG) ──────────
    force_opt_reg: float = 1e-4     # FORCE_OPT_REG

    # ── Torque Generator (config.hpp: WBT_W_DDQ, WBT_W_TAU) ─
    wbt_w_ddq: float = 1e-6         # WBT_W_DDQ
    wbt_w_tau: float = 1e-4         # WBT_W_TAU

    # ── 마찰 (config.hpp: FRICTION_MU) ───────────────────────
    friction_mu: float = 1.0        # FRICTION_MU

    # ── Preview Control (ZMPPreviewTrajectory용) ─────────────
    preview_horizon: int   = 1000   # Riccati preview 샘플 수
    preview_Qe: float     = 1.0    # 적분기 가중치
    preview_R: float      = 1e-6   # 입력 가중치

    # ── 파생 속성 ─────────────────────────────────────────────
    @property
    def omega(self) -> float:
        """ω = √(g / z_c)"""
        return (self.gravity / self.com_height) ** 0.5

    @property
    def ssp_time(self) -> float:
        return self.step_time - self.dsp_time

    @property
    def samples_per_step(self) -> int:
        return round(self.step_time / self.dt)
