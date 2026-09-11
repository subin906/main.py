import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DAILY_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
)
MOVIES_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"
)


st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 흥행 예측기")
st.caption("KOBIS 영화별 데이터를 이용한 다중 회귀 기반 총 관객 수 예측")


# ---------------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")

    # 컬럼명은 사용자가 제시한 이름을 기준으로 하되,
    # 혹시 CSV에 BOM/공백이 섞여 있어도 동작하도록 정리한다.
    daily.columns = daily.columns.astype(str).str.strip()
    movies.columns = movies.columns.astype(str).str.strip()

    # 영화코드는 두 파일에서 같은 의미이므로 문자열로 통일
    daily["영화코드"] = daily["영화코드"].astype(str).str.strip()
    movies["movieCd"] = movies["movieCd"].astype(str).str.strip()

    # 날짜를 날짜형으로 변환
    daily["날짜_dt"] = pd.to_datetime(
        daily["날짜"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    return daily, movies


try:
    daily, movies = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# ---------------------------------------------------------------------
# 기준 기간
# ---------------------------------------------------------------------
valid_dates = daily["날짜_dt"].dropna()

if len(valid_dates) > 0:
    period_start = valid_dates.min()
    period_end = valid_dates.max()

    st.info(
        f"**기준 기간:** {period_start.strftime('%Y-%m-%d')} ~ "
        f"{period_end.strftime('%Y-%m-%d')}"
    )
else:
    st.warning("일별 데이터에서 유효한 날짜를 찾을 수 없습니다.")
    period_start = period_end = None


# ---------------------------------------------------------------------
# 영화별 표 원본 표시
# ---------------------------------------------------------------------
st.subheader("영화별 데이터")

# 요청대로 영화코드 순으로 정렬
movies = movies.sort_values("movieCd").reset_index(drop=True)

# "영화별 표의 맨 위 열 줄" = 원본 표의 첫 번째 데이터 행
# 컬럼명 자체도 표에 그대로 보이도록 dataframe으로 표시한다.
st.dataframe(
    movies,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"영화별 표 전체 영화 수: **{len(movies):,}편**"
)


# ---------------------------------------------------------------------
# 모델용 데이터 구성
# ---------------------------------------------------------------------
# 숫자로 사용 가능한 후보 변수.
# total_audi는 목표값이므로 설명변수에서는 제외한다.
numeric_candidates = [
    "first_scrn",
    "first_show",
    "first_week_audi",
    "days_in_top10",
    "peak",
]

categorical_candidates = [
    "genre",
    "nation",
]

date_candidates = [
    "openDt",
    "first_date",
]

# 실제 데이터에 존재하는 변수만 사용
numeric_candidates = [c for c in numeric_candidates if c in movies.columns]
categorical_candidates = [c for c in categorical_candidates if c in movies.columns]
date_candidates = [c for c in date_candidates if c in movies.columns]


# 날짜는 회귀모델에 직접 문자열로 넣지 않고 날짜에서 숫자 특성을 만든다.
model_df = movies.copy()

for col in date_candidates:
    parsed = pd.to_datetime(model_df[col], format="%Y%m%d", errors="coerce")

    # 날짜를 ordinal 형태의 연속형 숫자로 변환
    model_df[f"{col}_ordinal"] = parsed.map(
        lambda x: x.toordinal() if pd.notna(x) else np.nan
    )

date_feature_candidates = [
    f"{c}_ordinal" for c in date_candidates
]


# ---------------------------------------------------------------------
# 변수 선택 UI
# ---------------------------------------------------------------------
st.subheader("예측 변수 선택")

st.write(
    "체크한 변수만 회귀 모델의 설명변수로 사용합니다. "
    "`total_audi`는 예측 대상이므로 선택할 수 없습니다."
)

selected_numeric = []
selected_categorical = []
selected_dates = []


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**숫자형 변수**")
    for col in numeric_candidates:
        label = {
            "first_scrn": "첫 관측일 스크린수",
            "first_show": "첫 관측일 상영횟수",
            "first_week_audi": "첫 주 관객",
            "days_in_top10": "10위권 유지일수",
            "peak": "성수기 개봉 여부",
        }.get(col, col)

        if st.checkbox(label, value=True, key=f"num_{col}"):
            selected_numeric.append(col)

with col2:
    st.markdown("**범주형 변수**")
    for col in categorical_candidates:
        label = {
            "genre": "장르",
            "nation": "국가",
        }.get(col, col)

        if st.checkbox(label, value=True, key=f"cat_{col}"):
            selected_categorical.append(col)

with col3:
    st.markdown("**날짜 변수**")
    for col in date_candidates:
        label = {
            "openDt": "개봉일",
            "first_date": "10위권 첫 등장일",
        }.get(col, col)

        if st.checkbox(label, value=True, key=f"date_{col}"):
            selected_dates.append(f"{col}_ordinal")


selected_features = (
    selected_numeric
    + selected_categorical
    + selected_dates
)

if not selected_features:
    st.warning("최소 한 개의 예측 변수를 선택해 주세요.")
    st.stop()


# ---------------------------------------------------------------------
# 목표값 정리
# ---------------------------------------------------------------------
model_df["total_audi"] = pd.to_numeric(
    model_df["total_audi"],
    errors="coerce",
)

# 영화별 표에 있는 영화는 모두 사용하되,
# 목표값 자체가 숫자로 해석되지 않는 경우 회귀 계산상 사용할 수 없으므로
# 해당 행은 평가/학습에서 제외한다.
usable = model_df["total_audi"].notna()

if usable.sum() < 10:
    st.error("회귀 모델을 학습하기에 유효한 영화 데이터가 너무 적습니다.")
    st.stop()

model_df = model_df.loc[usable].copy().reset_index(drop=True)


# ---------------------------------------------------------------------
# 10편마다 앞의 3편을 테스트용으로 분리
# ---------------------------------------------------------------------
# 이미 movieCd 순으로 정렬된 상태에서
# 0~9 -> 앞 3편 테스트, 나머지 7편 학습
# 10~19 -> 앞 3편 테스트, 나머지 7편 학습
# ...
# 마지막 블록이 10편보다 작아도 그 블록의 앞 3편을 테스트로 사용한다.
test_mask = np.zeros(len(model_df), dtype=bool)

for start in range(0, len(model_df), 10):
    end = min(start + 10, len(model_df))
    test_end = min(start + 3, end)
    test_mask[start:test_end] = True

train_df = model_df.loc[~test_mask].copy()
test_df = model_df.loc[test_mask].copy()


# ---------------------------------------------------------------------
# 회귀 파이프라인
# ---------------------------------------------------------------------
X_train = train_df[selected_features]
X_test = test_df[selected_features]

y_train = train_df["total_audi"]
y_test = test_df["total_audi"]


numeric_features = [
    c for c in selected_features
    if c in selected_numeric or c in selected_dates
]

categorical_features = [
    c for c in selected_features
    if c in selected_categorical
]

transformers = []

if numeric_features:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers.append(
        ("num", numeric_pipeline, numeric_features)
    )

if categorical_features:
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers.append(
        ("cat", categorical_pipeline, categorical_features)
    )


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)


# ---------------------------------------------------------------------
# 학습 및 평가
# ---------------------------------------------------------------------
try:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
except Exception as e:
    st.error(f"모델 학습 중 오류가 발생했습니다: {e}")
    st.stop()


# 음수 예측은 관객 수라는 물리적 의미에 맞지 않으므로 0으로 제한
predictions = np.maximum(predictions, 0)


r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# 평균 절대 백분율 오차.
# 실제 관객이 0인 행은 분모 문제 때문에 제외한다.
nonzero_actual = y_test.to_numpy() != 0

if nonzero_actual.any():
    mape = np.mean(
        np.abs(
            (
                y_test.to_numpy()[nonzero_actual]
                - predictions[nonzero_actual]
            )
            / y_test.to_numpy()[nonzero_actual]
        )
    ) * 100
else:
    mape = np.nan


# ---------------------------------------------------------------------
# 결과 요약
# ---------------------------------------------------------------------
st.subheader("모델 평가 결과")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("학습에 사용한 영화", f"{len(train_df):,}편")
m2.metric("평가한 영화", f"{len(test_df):,}편")
m3.metric("R² 점수", f"{r2:.4f}")
m4.metric("MAE", f"{mae:,.0f}명")
m5.metric(
    "MAPE",
    "계산 불가" if np.isnan(mape) else f"{mape:.1f}%",
)

st.caption(
    f"평가 기준 기간: "
    f"{period_start.strftime('%Y-%m-%d') if period_start is not None else '-'}"
    f" ~ "
    f"{period_end.strftime('%Y-%m-%d') if period_end is not None else '-'}"
)


# ---------------------------------------------------------------------
# 테스트 영화별 실제값 / 예측값
# ---------------------------------------------------------------------
result_df = test_df[
    ["movieCd", "movieNm", "total_audi"]
].copy()

result_df["predicted_total_audi"] = predictions
result_df["error"] = (
    result_df["predicted_total_audi"]
    - result_df["total_audi"]
)
result_df["absolute_error"] = result_df["error"].abs()

result_df = result_df.sort_values("movieCd").reset_index(drop=True)

st.subheader("테스트 영화별 예측 결과")

display_result = result_df.rename(
    columns={
        "movieCd": "영화코드",
        "movieNm": "영화명",
        "total_audi": "실제 총 관객 수",
        "predicted_total_audi": "예측 총 관객 수",
        "error": "예측 오차(예측-실제)",
        "absolute_error": "절대 오차",
    }
)

st.dataframe(
    display_result.style.format(
        {
            "실제 총 관객 수": "{:,.0f}",
            "예측 총 관객 수": "{:,.0f}",
            "예측 오차(예측-실제)": "{:+,.0f}",
            "절대 오차": "{:,.0f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------------------
# 1,000명 미만 예측 영화
# ---------------------------------------------------------------------
low_prediction_mask = result_df["predicted_total_audi"] < 1000
low_prediction_count = int(low_prediction_mask.sum())

st.subheader("1,000명 미만 예측")

st.write(
    f"예측 총 관객 수가 **1,000명 미만**인 영화는 "
    f"**{low_prediction_count}편**입니다."
)

if low_prediction_count > 0:
    low_df = result_df.loc[
        low_prediction_mask,
        ["movieCd", "movieNm", "total_audi", "predicted_total_audi"],
    ].copy()

    low_df.columns = [
        "영화코드",
        "영화명",
        "실제 총 관객 수",
        "예측 총 관객 수",
    ]

    st.dataframe(
        low_df.style.format(
            {
                "실제 총 관객 수": "{:,.0f}",
                "예측 총 관객 수": "{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------
# 산점도
# ---------------------------------------------------------------------
st.subheader("실제 관객 수 vs 예측 관객 수")

plot_df = result_df.copy()

# 로그축에서 0을 표시할 수 없으므로 실제/예측 모두 최소 1로 보정.
# 원본 결과값 자체는 변경하지 않는다.
plot_actual = np.maximum(
    plot_df["total_audi"].to_numpy(dtype=float),
    1,
)
plot_predicted = np.maximum(
    plot_df["predicted_total_audi"].to_numpy(dtype=float),
    1,
)

fig = go.Figure()

# 테스트 영화 산점도
fig.add_trace(
    go.Scatter(
        x=plot_actual,
        y=plot_predicted,
        mode="markers",
        marker=dict(
            size=9,
            color="#4C78A8",
            opacity=0.75,
        ),
        text=plot_df["movieNm"],
        customdata=plot_df[
            ["movieCd", "total_audi", "predicted_total_audi"]
        ].to_numpy(),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "영화코드: %{customdata[0]}<br>"
            "실제: %{customdata[1]:,.0f}명<br>"
            "예측: %{customdata[2]:,.0f}명"
            "<extra></extra>"
        ),
        name="테스트 영화",
    )
)

# 실제값 = 예측값 대각선
all_values = np.concatenate(
    [
        plot_actual,
        plot_predicted,
    ]
)

axis_min = max(1, np.nanmin(all_values))
axis_max = max(axis_min, np.nanmax(all_values))

fig.add_trace(
    go.Scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode="lines",
        line=dict(
            color="#E45756",
            width=2,
            dash="dash",
        ),
        name="실제 = 예측",
        hoverinfo="skip",
    )
)


# 예측값 1,000명 미만 영화는 그래프 바닥에 붙여 표시
low_plot = plot_df[
    plot_df["predicted_total_audi"] < 1000
].copy()

if not low_plot.empty:
    low_actual = np.maximum(
        low_plot["total_audi"].to_numpy(dtype=float),
        1,
    )

    # 로그축의 최저 영역에 붙인다.
    floor_y = axis_min

    fig.add_trace(
        go.Scatter(
            x=low_actual,
            y=np.full(len(low_actual), floor_y),
            mode="markers",
            marker=dict(
                size=12,
                color="#F58518",
                symbol="diamond",
                line=dict(
                    color="white",
                    width=1,
                ),
            ),
            text=low_plot["movieNm"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "실제: %{x:,.0f}명<br>"
                "예측: 1,000명 미만"
                "<extra></extra>"
            ),
            name="예측 1,000명 미만",
        )
    )


fig.update_layout(
    xaxis=dict(
        title="실제 총 관객 수",
        type="log",
    ),
    yaxis=dict(
        title="예측 총 관객 수",
        type="log",
    ),
    height=650,
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=60, r=30, t=70, b=60),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ---------------------------------------------------------------------
# 선택한 변수 및 데이터 분할 설명
# ---------------------------------------------------------------------
with st.expander("모델에 사용된 변수와 평가 방법"):
    st.write("**선택한 변수**")

    selected_labels = []

    label_map = {
        "first_scrn": "첫 관측일 스크린수",
        "first_show": "첫 관측일 상영횟수",
        "first_week_audi": "첫 주 관객",
        "days_in_top10": "10위권 유지일수",
        "peak": "성수기 개봉 여부",
        "genre": "장르",
        "nation": "국가",
        "openDt_ordinal": "개봉일",
        "first_date_ordinal": "10위권 첫 등장일",
    }

    for feature in selected_features:
        selected_labels.append(
            label_map.get(feature, feature)
        )

    for label in selected_labels:
        st.write(f"- {label}")

    st.write(
        """
**데이터 분할 방법**

영화별 표를 `movieCd` 순으로 정렬한 뒤 10편씩 묶었습니다.
각 묶음에서 앞의 3편을 테스트 데이터로, 나머지를 학습 데이터로 사용합니다.
마지막 묶음이 10편보다 적은 경우에도 그 묶음의 앞 3편을 테스트 데이터로 사용합니다.

테스트 영화는 모델 학습에 사용하지 않았기 때문에,
화면의 점수와 실제값-예측값 비교는 모두 학습에 사용하지 않은 영화에 대한 결과입니다.
"""
    )
