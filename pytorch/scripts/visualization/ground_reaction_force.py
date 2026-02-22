"""
Ground Reaction Force (GRF) 시각화 모듈.

Isaac Lab ContactSensor를 이용한 지면 반력 설정/조회/출력.
run_issac_lab.py에서 import하여 사용.

사용 예:
    from visualization.ground_reaction_force import (
        create_contact_sensor, GRFLogger,
    )
    sensor = create_contact_sensor("/World/G1")
    grf_logger = GRFLogger()
    ...
    grf_logger.update(sensor)
    grf_logger.print_summary()
"""

from __future__ import annotations
import torch
from typing import Optional


def create_contact_sensor(
    robot_prim_path: str = "/World/G1",
    foot_link_pattern: str = ".*_ankle_roll_link",
    history_length: int = 3,
    track_air_time: bool = True,
    debug_vis: bool = True,
):
    """ContactSensor 생성.

    Args:
        robot_prim_path: 로봇 USD prim 경로.
        foot_link_pattern: 발 링크 regex 패턴.
        history_length: 접촉 이력 길이.
        track_air_time: 공중 시간 추적 여부.
        debug_vis: 시각화 화살표 표시 여부.

    Returns:
        ContactSensor 인스턴스.
    """
    from isaaclab.sensors import ContactSensor, ContactSensorCfg

    cfg = ContactSensorCfg(
        prim_path=f"{robot_prim_path}/{foot_link_pattern}",
        history_length=history_length,
        track_air_time=track_air_time,
        debug_vis=debug_vis,
    )
    return ContactSensor(cfg)


class GRFLogger:
    """지면 반력 로거 — 접촉 센서 데이터 수집 및 출력.

    매 스텝 update()로 데이터 축적,
    print_summary()로 최근 통계 출력,
    print_detail()로 개별 바디 힘/air_time 출력.
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        # 이력: list of (fz_right, fz_left, total_fz)
        self.fz_history: list[tuple[float, float, float]] = []
        self.air_time_history: list[tuple[float, float]] = []  # (right, left)
        self.contact_count = 0

    def update(self, contact_sensor) -> Optional[dict]:
        """센서 데이터 수집. 매 sim step 후 호출.

        Returns:
            dict with 'net_forces', 'air_time' or None if sensor unavailable.
        """
        try:
            data = contact_sensor.data
            net_forces = data.net_forces_w          # (B, n_bodies, 3)
            air_time = data.current_air_time        # (B, n_bodies)

            # batch 0 기준, body 0=right, 1=left (prim_path 패턴 순서)
            n_bodies = net_forces.shape[1]
            if n_bodies >= 2:
                fz_r = net_forces[0, 0, 2].item()
                fz_l = net_forces[0, 1, 2].item()
                at_r = air_time[0, 0].item() if air_time is not None else 0.0
                at_l = air_time[0, 1].item() if air_time is not None else 0.0
            elif n_bodies == 1:
                fz_r = net_forces[0, 0, 2].item()
                fz_l = 0.0
                at_r = air_time[0, 0].item() if air_time is not None else 0.0
                at_l = 0.0
            else:
                return None

            total_fz = fz_r + fz_l
            self.fz_history.append((fz_r, fz_l, total_fz))
            self.air_time_history.append((at_r, at_l))
            self.contact_count += 1

            # 이력 제한
            if len(self.fz_history) > self.max_history:
                self.fz_history = self.fz_history[-self.max_history:]
                self.air_time_history = self.air_time_history[-self.max_history:]

            return {
                "net_forces": net_forces,
                "air_time": air_time,
                "fz_right": fz_r,
                "fz_left": fz_l,
                "total_fz": total_fz,
            }
        except Exception:
            return None

    def print_summary(self, last_n: int = 100):
        """최근 N 스텝 GRF 통계 출력."""
        if not self.fz_history:
            print("  [GRF] No data yet")
            return

        recent = self.fz_history[-last_n:]
        fz_r_vals = [x[0] for x in recent]
        fz_l_vals = [x[1] for x in recent]
        total_vals = [x[2] for x in recent]

        print(f"  [GRF] last {len(recent)} steps:")
        print(f"    RF fz: avg={sum(fz_r_vals)/len(fz_r_vals):+.2f}  "
              f"min={min(fz_r_vals):+.2f}  max={max(fz_r_vals):+.2f}")
        print(f"    LF fz: avg={sum(fz_l_vals)/len(fz_l_vals):+.2f}  "
              f"min={min(fz_l_vals):+.2f}  max={max(fz_l_vals):+.2f}")
        print(f"    Total: avg={sum(total_vals)/len(total_vals):+.2f}  "
              f"min={min(total_vals):+.2f}  max={max(total_vals):+.2f}")

        # air time
        recent_at = self.air_time_history[-last_n:]
        at_r = [x[0] for x in recent_at]
        at_l = [x[1] for x in recent_at]
        print(f"    Air time RF: {at_r[-1]:.3f}s  LF: {at_l[-1]:.3f}s")

    def print_detail(self, contact_sensor):
        """개별 바디별 접촉력 상세 출력."""
        try:
            net_forces = contact_sensor.data.net_forces_w  # (B, n_bodies, 3)
            air_time = contact_sensor.data.current_air_time
            force_history = contact_sensor.data.force_matrix_w  # (B, n_bodies, history, 3)

            print(f"  [GRF Detail]")
            for i in range(net_forces.shape[1]):
                f = net_forces[0, i]
                at = air_time[0, i].item() if air_time is not None else 0.0
                f_mag = f.norm().item()
                in_contact = f[2].item() > 1.0  # fz > 1N → 접촉 중

                label = "RF" if i == 0 else "LF" if i == 1 else f"body[{i}]"
                status = "CONTACT" if in_contact else "AIR"
                print(f"    {label} [{status}]: "
                      f"fx={f[0]:+8.2f} fy={f[1]:+8.2f} fz={f[2]:+8.2f}  "
                      f"|F|={f_mag:.2f}  air_t={at:.3f}s")

                # 이력 (최근 3 프레임)
                if force_history is not None and force_history.shape[2] > 1:
                    for h in range(min(3, force_history.shape[2])):
                        fh = force_history[0, i, h]
                        print(f"      hist[{h}]: fz={fh[2]:+8.2f}")
        except Exception as e:
            print(f"  [GRF Detail] Error: {e}")

    def reset(self):
        """이력 초기화."""
        self.fz_history.clear()
        self.air_time_history.clear()
        self.contact_count = 0
