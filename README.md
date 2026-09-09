import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 기온 예측기")
st.write("서울 기온 데이터를 이용해 연도별 평균기온의 추세를 분석하고, 선택한 연도의 예상 평균기온을 보여줍니다.")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"서울 기온 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 수업 기준 기간: 2025년까지.
# 관측일이 300일 미만인 해는 제외한다.
yearly = (
    df[df["연도"] <= 2025]
    .groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

yearly = yearly[yearly["관측일수"] >= 300].copy()
yearly = yearly.sort_values("연도")

if len(yearly) < 2:
    st.error("회귀 직선을 만들 수 있는 연도별 데이터가 충분하지 않습니다.")
    st.stop()

# 단순 선형회귀: 연도 -> 연평균기온
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
correlation = np.corrcoef(x, y)[0, 1]

yearly["회귀예상기온"] = slope * yearly["연도"] + intercept

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
n_years = len(yearly)

# 슬라이더
selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

predicted_temp = slope * selected_year + intercept

st.subheader(f"{selected_year}년 예상 평균기온")
st.metric("예상 기온", f"{predicted_temp:.2f} °C")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("회귀에 사용한 연도 수", f"{n_years}개")
with col2:
    st.metric("시작 연도", f"{start_year}년")
with col3:
    st.metric("끝 연도", f"{end_year}년")

st.write(f"**상관계수:** {correlation:.4f}")
st.caption(
    f"관측일이 300일 이상인 {start_year}~{end_year}년 자료만 사용했습니다. "
    f"2025년 이후 자료는 제외했습니다."
)

# 산점도 + 회귀 직선
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        text=[f"{int(yr)}년<br>관측일수: {int(days)}일"
              for yr, days in zip(yearly["연도"], yearly["관측일수"])],
        hovertemplate="%{text}<br>평균기온: %{y:.2f} °C<extra></extra>",
        marker=dict(size=7)
    )
)

# 회귀 직선은 1900~2100까지 표시하여 예측 구간도 확인할 수 있게 한다.
line_x = np.array([1900, 2100], dtype=float)
line_y = slope * line_x + intercept

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3)
    )
)

# 선택한 연도의 예측값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=13, symbol="star")
    )
)

fig.update_layout(
    title="연도별 평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title="구분",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("회귀식")
st.write(
    f"**예상 평균기온 = {slope:.6f} × 연도 + {intercept:.2f}**"
)
st.caption(
    "슬라이더에서 1900~2100년 중 원하는 연도를 선택하면, "
    "회귀 직선을 이용한 예상 평균기온이 계산됩니다."
)

with st.expander("분석에 사용한 연도별 데이터 보기"):
    display_df = yearly.copy()
    display_df["연평균기온"] = display_df["연평균기온"].round(2)
    display_df["회귀예상기온"] = display_df["회귀예상기온"].round(2)
    st.dataframe(display_df, use_container_width=True)
