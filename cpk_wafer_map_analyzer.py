"""
6 Sigma BB 기반 Cpk & Wafer Map 산포 분석기
=============================================

⚠️ 데이터 출처 안내
이 모듈은 실제 fab 계측 데이터가 아니라, 정규분포 기반으로 생성한 **합성(synthetic) 데이터**로
동작합니다. 6 Sigma Black Belt / Green Belt 교육에서 배운 Cpk(공정능력지수) 계산 로직과,
웨이퍼 내 위치에 따른 산포 시각화 방법론을 검증하기 위한 용도입니다.
실제 계측 데이터를 CSV로 업로드하면 동일한 로직으로 바로 분석할 수 있도록 설계했습니다.

Cpk 공식:
    Cpk = min( (USL - mean) / (3 * sigma), (mean - LSL) / (3 * sigma) )

일반적으로 반도체 공정에서는 Cpk >= 1.33 (4시그마 이상 여유)을 관리 기준으로 삼는다.

실행 방법:
    pip install streamlit numpy pandas matplotlib
    streamlit run cpk_wafer_map_analyzer.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import urllib.request
import os

st.set_page_config(page_title="Cpk & Wafer Map Analyzer", layout="wide")


def setup_korean_font():
    """OS별로 한글이 깨지지 않도록 matplotlib 폰트를 설정한다."""
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":  # macOS
        plt.rcParams["font.family"] = "AppleGothic"
    else:
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

st.title("📐 6 Sigma 기반 Cpk & Wafer Map 산포 분석기")
st.warning(
    "⚠️ **데이터 출처**: 이 데모는 정규분포 기반 **합성 데이터**로 동작합니다. "
    "실제 fab 계측 데이터가 아닙니다. 아래에서 직접 CSV를 업로드하면 실측 데이터로도 분석 가능합니다."
)

# -----------------------------------------------------------------------
# 1. Cpk 계산기
# -----------------------------------------------------------------------
st.header("1. 공정능력지수 (Cpk) 자동 산출")

with st.expander("📖 Cpk란? (왜 이 지표를 쓰는가)", expanded=False):
    st.markdown(
        r"""
Cpk는 공정이 규격(USL/LSL) 안에서 얼마나 안정적으로 관리되는지를 나타내는 지표다.
단순 평균/표준편차만으로는 "규격을 벗어날 위험"을 알 수 없기 때문에,
평균이 규격 중심에서 얼마나 치우쳐 있는지(치우침, k)까지 반영한 지표가 필요하다.

$$Cpk = \min\left(\frac{USL - \mu}{3\sigma},\ \frac{\mu - LSL}{3\sigma}\right)$$

- **Cpk >= 1.33**: 통상적인 반도체 공정 관리 기준 (약 4시그마 여유)
- **Cpk >= 1.67**: 고신뢰성 공정 기준 (약 5시그마 여유)
- **Cpk < 1.0**: 규격 이탈 위험이 높아 공정 개선이 필요한 상태
"""
    )

data_source = st.radio(
    "데이터 소스 선택", ["합성 데이터 생성 (데모)", "CSV 업로드 (실측 데이터)"], horizontal=True
)

if data_source == "합성 데이터 생성 (데모)":
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        n_samples = st.number_input("샘플 수", min_value=10, value=200, step=10)
    with g2:
        process_mean = st.number_input("공정 평균 (예: 두께 nm)", value=200.0, step=1.0)
    with g3:
        process_std = st.number_input("공정 표준편차", min_value=0.01, value=8.0, step=0.5)
    with g4:
        seed = st.number_input("랜덤 시드", min_value=0, value=42, step=1)

    rng = np.random.default_rng(int(seed))
    measurements = rng.normal(process_mean, process_std, size=int(n_samples))
    df = pd.DataFrame({"measurement": measurements})
    st.caption("↑ 정규분포 N(공정평균, 표준편차²)에서 샘플링한 합성 데이터입니다.")
else:
    uploaded = st.file_uploader("계측값 CSV 업로드 (단일 수치 컬럼)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        col_name = st.selectbox("분석할 컬럼 선택", df.columns)
        measurements = df[col_name].dropna().values
    else:
        st.info("CSV를 업로드하면 분석이 시작됩니다.")
        st.stop()

s1, s2 = st.columns(2)
with s1:
    usl = st.number_input("USL (규격 상한)", value=220.0, step=1.0)
with s2:
    lsl = st.number_input("LSL (규격 하한)", value=180.0, step=1.0)

mean = np.mean(measurements)
std = np.std(measurements, ddof=1)

cpu = (usl - mean) / (3 * std)
cpl = (mean - lsl) / (3 * std)
cpk = min(cpu, cpl)
cp = (usl - lsl) / (6 * std)

out_of_spec = ((measurements > usl) | (measurements < lsl)).sum()
out_of_spec_pct = out_of_spec / len(measurements) * 100

st.markdown("### 📊 분석 결과")
m1, m2, m3, m4 = st.columns(4)
m1.metric("평균 (μ)", f"{mean:.2f}")
m2.metric("표준편차 (σ)", f"{std:.2f}")
m3.metric("Cp", f"{cp:.3f}")
m4.metric("Cpk", f"{cpk:.3f}")

if cpk >= 1.67:
    st.success(f"✅ Cpk {cpk:.3f} — 고신뢰성 공정 기준(1.67) 충족")
elif cpk >= 1.33:
    st.success(f"✅ Cpk {cpk:.3f} — 일반 관리 기준(1.33) 충족")
elif cpk >= 1.0:
    st.warning(f"⚠️ Cpk {cpk:.3f} — 관리 기준(1.33) 미달, 공정 산포/치우침 개선 필요")
else:
    st.error(f"🚨 Cpk {cpk:.3f} — 규격 이탈 위험이 높은 상태 (1.0 미만)")

st.write(f"규격 이탈 샘플: **{out_of_spec}건 / {len(measurements)}건 ({out_of_spec_pct:.2f}%)**")

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(measurements, bins=30, color="steelblue", alpha=0.75, edgecolor="white")
ax.axvline(usl, color="red", linestyle="--", label=f"USL = {usl}")
ax.axvline(lsl, color="red", linestyle="--", label=f"LSL = {lsl}")
ax.axvline(mean, color="black", linestyle="-", label=f"Mean = {mean:.2f}")
ax.set_xlabel("측정값")
ax.set_ylabel("빈도")
ax.set_title("공정 분포 및 규격 (USL/LSL)")
ax.legend()
st.pyplot(fig)

st.divider()

# -----------------------------------------------------------------------
# 2. Wafer Map 산포 Heatmap
# -----------------------------------------------------------------------
st.header("2. 웨이퍼 내 영역별 산포 Heatmap")

with st.expander("📖 시뮬레이션 로직 설명", expanded=False):
    st.markdown(
        """
실제 fab 공정에서는 웨이퍼 중심부와 가장자리(edge) 간에 증착/식각 균일도 차이가 흔히 발생한다
(가스 유동, 플라즈마 밀도, 온도 분포의 반경 방향 편차 등). 이를 정성적으로 재현하기 위해,
**웨이퍼 중심에서 가장자리로 갈수록 값이 체계적으로 변하는 방사형(radial) 패턴 + 랜덤 노이즈**를
합성 데이터로 생성했다. 실제 계측 좌표 데이터가 있다면 동일한 방식으로 heatmap을 그릴 수 있다.
"""
    )

w1, w2, w3 = st.columns(3)
with w1:
    wafer_radius = st.slider("웨이퍼 반경 (mm)", 50, 150, 100, 5)
with w2:
    edge_effect_pct = st.slider("Edge 효과 강도 (중심 대비 %)", -30, 30, 12, 1,
                                 help="양수: 가장자리가 두꺼움/높음, 음수: 가장자리가 얇음/낮음")
with w3:
    noise_level = st.slider("랜덤 노이즈 수준 (%)", 0, 20, 5, 1)

grid_res = 80
x = np.linspace(-wafer_radius, wafer_radius, grid_res)
y = np.linspace(-wafer_radius, wafer_radius, grid_res)
xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx**2 + yy**2)
mask = r <= wafer_radius

radial_component = (r / wafer_radius) * (edge_effect_pct / 100) * process_mean if data_source == "합성 데이터 생성 (데모)" else (r / wafer_radius) * (edge_effect_pct / 100) * mean
base_value = process_mean if data_source == "합성 데이터 생성 (데모)" else mean

rng2 = np.random.default_rng(7)
noise = rng2.normal(0, base_value * noise_level / 100, size=xx.shape)

wafer_values = base_value + radial_component + noise
wafer_values_masked = np.where(mask, wafer_values, np.nan)

fig2, ax2 = plt.subplots(figsize=(6.5, 6))
im = ax2.imshow(
    wafer_values_masked, extent=(-wafer_radius, wafer_radius, -wafer_radius, wafer_radius),
    origin="lower", cmap="RdYlBu_r"
)
circle = plt.Circle((0, 0), wafer_radius, fill=False, color="black", linewidth=1.5)
ax2.add_patch(circle)
ax2.set_xlabel("X (mm)")
ax2.set_ylabel("Y (mm)")
ax2.set_title("웨이퍼 내 영역별 산포 (합성 데이터)")
ax2.set_aspect("equal")
plt.colorbar(im, ax=ax2, label="측정값")
st.pyplot(fig2)

center_val = base_value
edge_val = base_value * (1 + edge_effect_pct / 100)
st.metric("중심 대비 가장자리 편차", f"{edge_val - center_val:+.2f} ({edge_effect_pct:+d}%)")

st.divider()
st.caption(
    "본 모듈은 6 Sigma Black/Green Belt 교육에서 학습한 Cpk 계산 방법론을 검증하기 위해 "
    "합성 데이터로 구현했습니다. 실제 계측 데이터 CSV를 업로드하면 동일한 로직으로 분석할 수 있습니다."
)
