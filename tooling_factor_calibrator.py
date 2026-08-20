"""
PVD / Thin-Film Tooling Factor 자동 캘리브레이션 계산기
=====================================================

배경 (실제 경험 기반):
E-beam Evaporator / Thermal Evaporator 증착 시 QCM(Quartz Crystal Microbalance)이
표시하는 두께는 "Tooling Factor"라는 보정 상수에 의존한다. 이 상수가 실제 지형(Tooling)과
어긋나 있으면, QCM이 보여주는 두께와 실제 증착된 두께(SEM 단면 / AFM 스텝하이트로 실측)가
어긋난다.

보정 공식:
    New Tooling Factor = Old Tooling Factor x (실측 두께 / QCM 표시 두께)

실제 사례: QCM 기준 Tooling Factor 1.211로 증착했으나 SEM 단면 실측 결과 목표 200nm 대비
과증착이 확인되어, 위 공식으로 재계산한 결과 Tooling Factor가 1.813으로 보정됨
(비율 1.813 / 1.211 ≈ 1.497, 즉 SEM 실측 두께가 QCM 표시 두께보다 약 49.7% 더 두꺼웠던 셈).

이 앱은 그 보정 과정을 일반화한 계산기이며, 추가로 증착 속도(Rate) 변동에 따른
두께 오차율 시뮬레이션을 제공한다 (Rate가 불안정할수록 QCM 응답 지연/소스 저항 변화로 인한
오차가 커진다는 실무 경험을 반영).

실행 방법:
    pip install streamlit numpy pandas matplotlib
    streamlit run tooling_factor_calibrator.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import urllib.request
import os

st.set_page_config(page_title="Tooling Factor Calibrator", layout="wide")


def setup_korean_font():
    """OS별로 한글이 깨지지 않도록 matplotlib 폰트를 설정한다."""
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":  # macOS
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        # Streamlit Cloud 등 리눅스 환경: Noto Sans KR을 다운받아 등록
        font_path = "/tmp/NotoSansKR-Regular.ttf"
        if not os.path.exists(font_path):
            try:
                url = (
                    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/"
                    "Korean/NotoSansCJKkr-Regular.otf"
                )
                urllib.request.urlretrieve(url, font_path)
            except Exception:
                pass
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False


setup_korean_font()

st.title("🔬 PVD Thin-Film Tooling Factor 자동 캘리브레이션 계산기")
st.caption(
    "QCM 표시 두께와 SEM/AFM 실측 두께의 오차를 이용해 Tooling Factor를 재산출합니다."
)

# -----------------------------------------------------------------------
# 1. Tooling Factor 계산기
# -----------------------------------------------------------------------
st.header("1. Tooling Factor 보정 계산")

with st.expander("📖 계산 로직 설명 (왜 이렇게 계산하는가)", expanded=False):
    st.markdown(
        r"""
QCM은 수정 진동자의 공진주파수 변화를 질량 변화로 환산해 두께를 표시하지만,
Tooling Factor(장비별 기하학적 배치·증착원-기판-센서 간 거리 차이를 보정하는 상수)가
부정확하면 표시 두께와 실제 두께가 어긋난다.

**보정 공식:**

$$TF_{new} = TF_{old} \times \frac{\text{실측 두께 (SEM/AFM)}}{\text{QCM 표시 두께}}$$

즉, QCM이 두께를 과소 표시했다면(실측이 더 두꺼웠다면) Tooling Factor를 키워서
다음 증착부터 QCM이 더 일찍(더 얇을 때) 증착을 멈추도록 보정한다.
"""
    )

col1, col2, col3 = st.columns(3)
with col1:
    target_thickness = st.number_input(
        "Target 두께 (nm)", min_value=0.0, value=200.0, step=1.0,
        help="공정에서 의도한 목표 두께"
    )
with col2:
    old_tf = st.number_input(
        "기존 Tooling Factor", min_value=0.0001, value=1.211, step=0.001, format="%.3f"
    )
with col3:
    qcm_indicated = st.number_input(
        "QCM 표시 두께 (nm)", min_value=0.0001, value=200.0, step=1.0,
        help="기존 Tooling Factor 적용 상태에서 QCM이 표시한 두께"
    )

actual_thickness = st.number_input(
    "실측 두께 - SEM 단면 / AFM 스텝하이트 (nm)",
    min_value=0.0001, value=299.4, step=0.1,
    help="증착 후 SEM 단면 또는 AFM으로 실제 측정한 두께"
)

new_tf = old_tf * (actual_thickness / qcm_indicated)
error_pct = (actual_thickness - qcm_indicated) / qcm_indicated * 100
tf_change_pct = (new_tf - old_tf) / old_tf * 100

st.markdown("### 📊 계산 결과")
r1, r2, r3 = st.columns(3)
r1.metric("보정된 Tooling Factor", f"{new_tf:.3f}", f"{tf_change_pct:+.1f}%")
r2.metric("QCM 대비 실측 오차율", f"{error_pct:+.1f}%")
r3.metric("Target 대비 실측 두께 오차", f"{(actual_thickness - target_thickness) / target_thickness * 100:+.1f}%")

if abs(error_pct) > 30:
    st.warning(
        f"⚠️ QCM-실측 오차가 {error_pct:.1f}%로 큽니다. "
        "센서 노후화, 소스-기판 거리 변경, 또는 툴링 인수 계산 시점의 물질 밀도/Z-ratio 설정을 재점검하세요."
    )
else:
    st.success("✅ 오차율이 통상 관리 범위(±30%) 내에 있습니다.")

st.markdown(
    f"""
> **검증 예시 (실제 경험 재현):** 기존 TF = 1.211, QCM 표시 200nm, SEM 실측 299.4nm 입력 시
> → 보정된 TF ≈ **1.813** (실제 인턴 경험에서 도출된 값과 일치)
"""
)

st.divider()

# -----------------------------------------------------------------------
# 2. 증착 속도(Rate) 변동에 따른 두께 오차율 시각화
# -----------------------------------------------------------------------
st.header("2. 증착 속도(Rate) 변동 → 두께 오차율 시뮬레이션")

with st.expander("📖 시뮬레이션 로직 설명", expanded=False):
    st.markdown(
        """
증착 속도(Rate)가 불안정하면 QCM의 응답 지연(response lag)과 소스 저항 변화(Joule heating,
Q=I²Rt)로 인한 rate spike가 겹쳐 실제 두께와 QCM 표시 두께 사이의 누적 오차가 커진다.
아래 시뮬레이션은 목표 Rate 대비 변동성(노이즈 수준)을 입력받아, 누적 두께 오차가
시간에 따라 어떻게 벌어지는지 보여준다. **실제 계측값이 아니라 정성적 경향을 보여주기 위한
합성 시뮬레이션**이다.
"""
    )

sc1, sc2, sc3 = st.columns(3)
with sc1:
    target_rate = st.slider("목표 증착 Rate (Å/s)", 0.5, 10.0, 2.0, 0.1)
with sc2:
    rate_noise_pct = st.slider("Rate 변동성 (표준편차, %)", 0, 50, 15, 1)
with sc3:
    deposition_time = st.slider("총 증착 시간 (s)", 10, 300, 100, 10)

np.random.seed(42)
time_steps = np.arange(0, deposition_time, 1)
noise = np.random.normal(0, target_rate * rate_noise_pct / 100, size=len(time_steps))
actual_rate = np.clip(target_rate + noise, 0, None)

ideal_thickness_curve = target_rate * time_steps
actual_thickness_curve = np.cumsum(actual_rate)

# t=0 근처는 이상적 두께가 0에 가까워 오차율(%)이 무의미하게 커진다.
# 두께가 목표 Rate 1초치 이상 쌓인 시점부터만 오차율을 계산해 그래프 왜곡을 방지한다.
min_valid_thickness = max(target_rate, 1e-3)
valid_mask = ideal_thickness_curve >= min_valid_thickness

thickness_error_pct = np.full_like(ideal_thickness_curve, np.nan)
thickness_error_pct[valid_mask] = (
    (actual_thickness_curve[valid_mask] - ideal_thickness_curve[valid_mask])
    / ideal_thickness_curve[valid_mask]
    * 100
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(time_steps, ideal_thickness_curve, label="이상적 두께 (목표 Rate 고정)", linestyle="--")
ax1.plot(time_steps, actual_thickness_curve, label="실제 누적 두께 (Rate 변동 반영)")
ax1.set_xlabel("시간 (s)")
ax1.set_ylabel("누적 두께 (Å)")
ax1.set_title("이상적 두께 vs 실제 누적 두께")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(time_steps, thickness_error_pct, color="firebrick")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xlabel("시간 (s)")
ax2.set_ylabel("두께 오차율 (%)")
ax2.set_title("시간에 따른 누적 두께 오차율")
ax2.grid(alpha=0.3)

st.pyplot(fig)

final_error = thickness_error_pct[-1]
if np.isnan(final_error):
    st.info("증착 시간이 너무 짧아 오차율을 계산할 수 없습니다. 슬라이더에서 증착 시간을 늘려주세요.")
else:
    st.metric("증착 종료 시점 누적 두께 오차율", f"{final_error:+.2f}%")

if rate_noise_pct >= 25:
    st.info(
        "💡 Rate 변동성이 25% 이상이면 누적 오차가 실무 관리 기준을 벗어나기 쉽습니다. "
        "Source 전류 제어(전류/증착률 관계, Q=I²Rt) 및 QCM 응답 안정화 구간 확보가 필요합니다."
    )

st.divider()
st.caption(
    "본 계산기는 실제 PVD/Thermal Evaporator 공정 경험(Ti/Au 전극 증착, "
    "Tooling Factor 1.211→1.813 보정 사례)을 일반화하여 구현했습니다. "
    "Rate 변동 시뮬레이션은 정성적 경향을 보여주기 위한 합성 데이터입니다."
)
