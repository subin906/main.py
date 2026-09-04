import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="영화 데이터 그래프 도감 2 - 분포와 관계",
    page_icon="🎬",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

st.title("영화 데이터 그래프 도감 2 - 분포와 관계")
st.caption("1년간 박스오피스 10위권에 든 영화 가운데 이 기간에 개봉한 216편의 요약표")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # 세로막대(|)로 여러 장르가 적힌 경우 첫 번째 장르만 사용
    df["genre"] = (
        df["genre"]
        .fillna("미상")
        .astype(str)
        .str.split("|")
        .str[0]
        .str.strip()
        .replace("", "미상")
    )
    return df

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결 또는 데이터 주소를 확인해 주세요.")
    st.exception(e)
    st.stop()

# ── 그래프 1: 장르별 영화 편수 ─────────────────────────────
st.subheader("1. 장르별 영화 편수")

genre_counts = (
    df["genre"]
    .value_counts()
    .rename_axis("장르")
    .reset_index(name="영화 편수")
)
genre_counts["비율"] = genre_counts["영화 편수"] / genre_counts["영화 편수"].sum() * 100

fig = px.pie(
    genre_counts,
    names="장르",
    values="영화 편수",
    hole=0.55,
    custom_data=["영화 편수", "비율"],
)
fig.update_traces(
    hovertemplate="<b>%{label}</b><br>편수: %{customdata[0]}편<br>비율: %{customdata[1]:.1f}%<extra></extra>",
    textinfo="label",
)
fig.update_layout(
    margin=dict(t=30, b=20, l=20, r=20),
    legend_title_text="장르",
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    <div style="
        border: 1px solid #d9d9d9;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 4px;
        margin-bottom: 24px;
        background-color: #fafafa;
    ">
        <b>이 그래프로 알 수 있는 것</b><br>
        어떤 장르의 영화가 이 기간의 박스오피스 10위권 영화에서 많이 등장했는지 한눈에 비교할 수 있습니다.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.info("다음 그래프를 추가할 때는 같은 형식으로 그래프 아래에 ‘이 그래프로 알 수 있는 것’ 영역을 만들면 됩니다.")
