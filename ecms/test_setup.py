import numpy as np, pytest, math
from powertrain_setup import make_vehicle_params

ENGINE_XLS = "2.0L SKYACTIV Engine LEV III Fuel_Fuel Map Data.xlsx"
MOTOR_XLS  = "30kW 270V EMOT_Electrical Power Consumption Data.xlsx"

def test_maps_and_params():
    p = make_vehicle_params(ENGINE_XLS, MOTOR_XLS)

    # ① BSFC 값 유효성 (Positive & not NaN)
    tq_test, rpm_test = 150, 2500
    bsfc_val = p.bsfc((tq_test, rpm_test))
    assert not math.isnan(bsfc_val) and bsfc_val > 50

    # ② 모터 효율 0~1 범위
    eta_val = p.mot_eta((50.0, 2000.0))
    assert 0.0 <= eta_val <= 1.0

    # ③ 배터리 Q 값 정확 변환
    assert pytest.approx(p.Q * p.U_oc / 3.6e6, rel=1e-6) == 2.0   # kWh 역변환

    # ④ 저항 파워 기본 테스트
    from powertrain_setup import resistive_power
    P = resistive_power(20.0, p)           # v=20 m/s (~72 km/h)
    assert P > 0


# test_setup.py 맨 끝에 추가
if __name__ == "__main__":
    from pprint import pprint
    try:
        test_maps_and_params()
        pprint("✅ All assertions passed")
    except AssertionError as e:
        pprint(f"❌ Test failed: {e}")
