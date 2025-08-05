from __future__ import annotations
import numpy as np, math
from typing import Tuple

# ───────────────────────────────────────────────────────── 기본 물리
def engine_speed_default(v: float, p):     # 고정 기어 대안 (fallback)
    return (v / p.wheel_radius) * p.gear_ratio * p.final_drive

def engine_speed_opt(Peng_W: float, v: float, p):
    """
    ① 요구 Peng 에 대해 rpm_opt_for_power(kW) 사용해 최적 rpm 근사
    ② 만약 차량속도에 따른 기계식 rpm < idle → idle 로 픽스
    """
    rpm_opt = float(p.rpm_opt_for_power(max(Peng_W / 1000, 0.1)))   # kW→rpm
    rpm_mech = engine_speed_default(v, p) * 60 / (2 * math.pi)
    return max(rpm_opt, rpm_mech, 800) * 2 * math.pi / 60           # rad/s

def fuel_flow(P_eng: float, w_eng: float, p):
    if P_eng <= 0 or w_eng < 1e-3: return 0.0
    tq = P_eng*1e3 / w_eng
    bsfc = float(p.bsfc((tq, w_eng)))
    return P_eng * bsfc / 3.6e6

# ───────────────────────────────────────────────────────── Peng(u) 등
def Peng_from_u(u, P_req, p):
    return p.Q**2 * p.R * u**2 + p.Q * p.U_oc * u + P_req

def optimal_u(P_req, lam, p):
    a, b = p.Q**2 * p.R, p.Q * p.U_oc
    u_star = (-b + lam*p.q_batt)/(2*a)
    Δ_dis = p.U_oc**2 - 4*p.R*(P_req - p.P_batt_max_dis)
    Δ_chg = p.U_oc**2 - 4*p.R*(P_req - p.P_batt_max_chg)
    u_max = (-p.U_oc + math.sqrt(max(Δ_dis,0)))/(2*p.Q*p.R)
    u_min = (-p.U_oc + math.sqrt(max(Δ_chg,0)))/(2*p.Q*p.R)
    return float(np.clip(u_star, u_min, u_max))

def ecms_step(P_req, v, soc, lam, p, dt=1.0):
    u = optimal_u(P_req, lam, p)
    Peng = Peng_from_u(u, P_req, p)
    w_eng = engine_speed_opt(Peng, v, p)               # ★ 최적 rpm 사용
    P_batt = P_req - Peng
    fuel = fuel_flow(Peng, w_eng, p) * dt
    soc_n = soc - u*dt
    return Peng, P_batt, u, soc_n, fuel

# ───────────────────────────────────────────────────────── 파워 프로파일
def calculate_power_profile(v_seq: np.ndarray, p, grade: float = 0.0):
    accel = np.diff(v_seq, prepend=v_seq[0]) / 1.0
    g = 9.81
    F_aero = 0.5 * p.rho_air * p.C_d * p.frontal_area * v_seq**2
    F_roll = p.mass * g * p.C_rr
    F_grade= p.mass * g * math.sin(math.atan(grade/100))
    F_inert= p.mass * accel
    return v_seq * (F_aero + F_roll + F_grade + F_inert)

# ───────────────────────────────────────────────────────── λ* (식34)
def solve_lambda_sequence(a,b,q,s0,st,dt):
    H_inv = np.diag(1/(2*a*dt))
    A = -np.ones_like(b)*q
    B = b*dt
    C = q*(st - s0)
    lam = (C - A@H_inv@B.T)/(A@H_inv@A.T)
    return float(lam)

# ───────────────────────────────────────────────────────── RH 시뮬
def run_receding_horizon_sim(
    v_true, soc0, p, horizon=30, dt=1.0,
    sigma_const: float = 1.0,
):
    N=len(v_true)
    Peng_hist=np.zeros(N); Pbat_hist=np.zeros(N); soc_hist=np.zeros(N); lam_hist=np.zeros(N)
    soc=soc0
    w = {'mean':0.6,'pess':0.2,'opt':0.2}

    for k in range(N):
        H=min(horizon,N-k)
        mu = v_true[k:k+H]
        sigma = np.full(H, sigma_const)
        scen = {
            'mean':mu,
            'pess':np.maximum(0,mu-sigma),
            'opt' :mu+sigma
        }
        lam_s = {key: solve_lambda_sequence(
                    a=np.full(H,p.Q**2*p.R),
                    b=np.full(H,p.Q*p.U_oc),
                    q=p.q_batt, s0=soc, st=soc, dt=dt)
                 for key,mu in scen.items()}
        P_req_now = calculate_power_profile(np.array([v_true[k]]), p)[0]
        u_s = {key: optimal_u(P_req_now, lam, p) for key,lam in lam_s.items()}
        u = w['mean']*u_s['mean'] + w['pess']*u_s['pess'] + w['opt']*u_s['opt']
        Peng = Peng_from_u(u,P_req_now,p)
        w_eng = engine_speed_opt(Peng, v_true[k], p)
        P_batt = P_req_now - Peng
        soc -= u*dt
        fuel = fuel_flow(Peng, w_eng, p)*dt

        Peng_hist[k]=Peng; Pbat_hist[k]=P_batt; soc_hist[k]=soc; lam_hist[k]=np.dot(list(w.values()), list(lam_s.values()))
    fuel_tot = fuel_flow(np.maximum(Peng_hist,0), engine_speed_opt(Peng_hist,v_true,p), p).sum()*dt
    return {"Peng_W":Peng_hist,"P_batt_W":Pbat_hist,"SOC":soc_hist,"lambda":lam_hist,"fuel_kg":fuel_tot}


# --- 사용 예시 ---
# if __name__ == '__main__':
#     # 1. 차량 파라미터 로드
#     p = VehicleParams()
#
#     # 2. 테스트용 주행 사이클 로드 (예: HWFET)
#     #    이것이 실제 도로에서 마주치는 'Ground Truth' 속도 프로파일이라고 가정
#     test_cycle = pd.read_csv('HWFET.csv')
#     v_actual_profile = test_cycle['speed_mps'].values
#
#     # 3. 가상의 예측기 정의 (실제로는 학습된 Transformer 모델 객체)
#     class DummyPredictor:
#         def predict(self, past_data):
#             # 이 부분에서 Transformer 모델이 예측을 수행
#             pass
#     predictor = DummyPredictor()
#
#     # 4. 시뮬레이션 실행
#     results = run_receding_horizon_simulation(
#         v_actual_profile=v_actual_profile,
#         predictor=predictor,
#         soc0=0.6,
#         p=p,
#         horizon=30
#     )
#
#     print(f"Total fuel consumed: {results['fuel_kg']:.3f} kg")