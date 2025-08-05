"""
powertrain_setup.py
  · load_maps(...)          : 엔진 BSFC·모터 효율 맵 로더
  · VehicleParams(@dataclass)
  · make_vehicle_params(...) : VehicleParams 객체 생성기
  · resistive_power(v, p)   : 구름·공력 저항 파워 계산

※ 외부 노출 API를 __all__ 에 명시해 두었으므로
   from powertrain_setup import *  시 꼭 필요한 함수·클래스만 import 됩니다.
"""
import pandas as pd, numpy as np
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator, interp1d
from typing import Callable, Tuple, List, Any

__all__ = [
    "load_maps",
    "VehicleParams",
    "make_vehicle_params",
    "resistive_power",
]

# ───────────────────────────────────────────────────────── load_maps
def load_maps(
    engine_file: str, motor_file: str
) -> Tuple[RegularGridInterpolator, Callable[[np.ndarray], np.ndarray], RegularGridInterpolator]:
    """Engine BSFC map, rpm_opt_for_power, Motor efficiency map 반환"""
    # --- Engine BSFC -------------------------------------------------------
    df_e = pd.read_excel(engine_file, header=None, skiprows=6, usecols=[0, 1, 3],
                         names=["RPM", "Torque", "BSFC"])
    df_e = df_e.dropna()
    rpm_e  = np.sort(df_e["RPM"].unique())
    tq_e   = np.sort(df_e["Torque"].unique())
    bsfc_mat = df_e.pivot(index="Torque", columns="RPM", values="BSFC").values
    bsfc_map = RegularGridInterpolator((tq_e, rpm_e), bsfc_mat,
                                       bounds_error=False, fill_value=None)

    # rpm_opt_for_power (최적 rpm ↔ 파워)
    RPM_mat, T_mat = np.meshgrid(rpm_e, tq_e, indexing="xy")
    P_mat = (T_mat * (RPM_mat * 2 * np.pi / 60)) / 1000      # [kW]
    P, rpm_flat = P_mat.flatten(), RPM_mat.flatten()
    valid = P > 0.1
    sort_idx = np.argsort(P[valid])
    uniq_P, uniq_idx = np.unique(P[valid][sort_idx], return_index=True)
    rpm_opt_for_power = interp1d(
        uniq_P, rpm_flat[valid][sort_idx][uniq_idx],
        bounds_error=False, fill_value="extrapolate"
    )

    # --- Motor η -----------------------------------------------------------
    df_m = pd.read_excel(motor_file, header=None, skiprows=5, usecols=[0, 1, 4],
                         names=["RPM", "Torque", "EffPct"]).dropna()
    rpm_m = np.sort(df_m["RPM"].unique())
    tq_m  = np.sort(df_m["Torque"].unique())
    eta_mat = (df_m.pivot(index="Torque", columns="RPM", values="EffPct").values) / 100.0
    mot_eta = RegularGridInterpolator((tq_m, rpm_m), eta_mat,
                                      bounds_error=False, fill_value=0.0)

    return bsfc_map, rpm_opt_for_power, mot_eta


# ───────────────────────────────────────────────────────── VehicleParams
@dataclass
class VehicleParams:
    # 배터리
    Q: float; R: float; U_oc: float
    # 맵
    bsfc: RegularGridInterpolator
    mot_eta: RegularGridInterpolator
    rpm_opt_for_power: Callable[[float], float]   # ★ 추가: kW → rpm
    # 드라이브라인
    gear_ratio: float; final_drive: float; wheel_radius: float; eta_driveline: float
    # 배터리 전력 제약
    P_batt_max_dis: float; P_batt_max_chg: float
    # 차체
    mass: float; wheelbase: float; C_rr: float; C_d: float; frontal_area: float
    rho_air: float = 1.225  # [kg/m³]

    @property
    def q_batt(self) -> float:        # [J]  (SOC→에너지 변환계수)
        return self.Q * self.U_oc


# ───────────────────────────────────────────────────────── make_vehicle_params
def make_vehicle_params(
    engine_file: str,
    motor_file: str,
    *,
    R_int: float = 0.015,      # Ω (필요 시 교체)
    cap_kwh: float = 2.0,      # kWh
    U_oc: float = 270.0        # V (정격 전압)
) -> VehicleParams:
    bsfc_map, rpm_opt, mot_eta = load_maps(engine_file, motor_file)

    Q_coul = cap_kwh * 3.6e6 / U_oc   # 정확 변환

    return VehicleParams(
        Q=Q_coul, R=R_int, U_oc=U_oc,
        bsfc=bsfc_map, mot_eta=mot_eta,rpm_opt_for_power=rpm_opt,
        gear_ratio=5.0, final_drive=3.3, wheel_radius=0.32, eta_driveline=0.95,
        P_batt_max_dis=80_000, P_batt_max_chg=-40_000,
        mass=1550.0, wheelbase=2.795, C_rr=0.015, C_d=0.28, frontal_area=2.2
    )


# ───────────────────────────────────────────────────────── resistive_power
def resistive_power(v_mps: float, p: VehicleParams) -> float:
    """구름 + 공력 저항 파워 [W] (경사·가속 제외)"""
    F_roll = p.mass * 9.81 * p.C_rr
    F_aero = 0.5 * p.rho_air * p.C_d * p.frontal_area * v_mps**2
    return (F_roll + F_aero) * v_mps
