import torch


def batched_skew_symmetric(vec: torch.Tensor) -> torch.Tensor:
    """
    (num_envs, 3) 형태의 벡터 배치를 입력받아
    (num_envs, 3, 3) 형태의 Skew-Symmetric 행렬 배치를 반환
    """
    num_envs = vec.shape[0]
    device = vec.device
    dtype = vec.dtype

    S = torch.zeros((num_envs, 3, 3), device=device, dtype=dtype)

    S[:, 0, 1] = -vec[:, 2]
    S[:, 0, 2] =  vec[:, 1]
    S[:, 1, 0] =  vec[:, 2]
    S[:, 1, 2] = -vec[:, 0]
    S[:, 2, 0] = -vec[:, 1]
    S[:, 2, 1] =  vec[:, 0]

    return S


# ── Cycloid XY 보간 ──────────────────────────────────────────
# C++ foot_trajectory.cpp의 cycloid 계산 대응
# θ = 2π·p, mod = (θ - sinθ) / (2π)

def cycloid_pos(p: torch.Tensor) -> torch.Tensor:
    """Cycloid 위치 보간 계수. p ∈ [0,1] → mod ∈ [0,1]"""
    theta = 2.0 * torch.pi * p
    return (theta - torch.sin(theta)) / (2.0 * torch.pi)


def cycloid_vel(p: torch.Tensor, ssp_time: float) -> torch.Tensor:
    """Cycloid 속도 계수. dc/dt = (1 - cosθ) / ssp"""
    theta = 2.0 * torch.pi * p
    return (1.0 - torch.cos(theta)) / ssp_time


def cycloid_acc(p: torch.Tensor, ssp_time: float) -> torch.Tensor:
    """Cycloid 가속도 계수. d²c/dt² = 2π sinθ / ssp²"""
    theta = 2.0 * torch.pi * p
    return 2.0 * torch.pi * torch.sin(theta) / (ssp_time * ssp_time)


# ── 5차 Bezier Z ─────────────────────────────────────────────
# C++ foot_trajectory.cpp의 bezierZ, bezierZ_ds, bezierZ_dds 대응
# 제어점: P = [gz, gz, gz+h, gz+h, gz, gz]
# 정리: B(s) = gz + h * 10 * s² * (1-s)²

def bezier_z_pos(s: torch.Tensor, gz: torch.Tensor, h: float) -> torch.Tensor:
    """B(s) = gz + h * 10 * s² * (1-s)²"""
    u = 1.0 - s
    return gz + h * 10.0 * s * s * u * u


def bezier_z_vel(s: torch.Tensor, h: float, ssp_time: float) -> torch.Tensor:
    """dB/dt = dB/ds * ds/dt, ds/dt = 1/ssp
    dB/ds = h * 20 * s * (1-s) * (1-2s)"""
    ds_dt = 1.0 / ssp_time
    return h * 20.0 * s * (1.0 - s) * (1.0 - 2.0 * s) * ds_dt


def bezier_z_acc(s: torch.Tensor, h: float, ssp_time: float) -> torch.Tensor:
    """d²B/dt² = d²B/ds² * (ds/dt)²
    d²B/ds² = h * 20 * (1 - 6s + 6s²)"""
    ds_dt = 1.0 / ssp_time
    return h * 20.0 * (1.0 - 6.0 * s + 6.0 * s * s) * ds_dt * ds_dt


# ── 접촉 상태 판별 ─────────────────────────────────────────────
# C++ getContactState() 대응

def get_contact_state(
    sim_time: float,
    batch_size: int,
    n_steps: int,
    step_time: float,
    dsp_time: float,
    device: "torch.device",
    dtype: "torch.dtype",
) -> torch.Tensor:
    """접촉 상태 판별 — C++ getContactState() 대응.

    Returns:
        contact: (B, 2) — [right, left], 1.0=접촉, 0.0=스윙
    """
    contact = torch.ones(batch_size, 2, device=device, dtype=dtype)

    t_acc = 0.0
    for i in range(n_steps):
        if sim_time < t_acc + step_time:
            t_in_step = sim_time - t_acc
            in_ssp = (t_in_step >= dsp_time) and (i + 1 < n_steps)
            if in_ssp:
                if i % 2 == 0:
                    contact[:, 0] = 0.0   # right swing
                else:
                    contact[:, 1] = 0.0   # left swing
            break
        t_acc += step_time

    return contact
