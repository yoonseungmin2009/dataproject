import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

st.set_page_config(
    page_title="서울 기후 변화 분석",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 스타일 ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700;900&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

.main { background: #0a0e1a; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1525 50%, #0a0e1a 100%);
    color: #e8eaf0;
}

.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px 28px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: rgba(255,120,80,0.4);
    transform: translateY(-2px);
}

.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    margin: 8px 0 4px;
}

.metric-label {
    font-size: 0.78rem;
    font-weight: 300;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8a8fa8;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8a8fa8;
    margin-bottom: 4px;
}

.badge-pre {
    display:inline-block;
    background:rgba(100,160,255,0.15);
    color:#64a0ff;
    border:1px solid rgba(100,160,255,0.3);
    border-radius:6px;
    padding:2px 10px;
    font-size:0.75rem;
    font-weight:700;
    letter-spacing:.08em;
}
.badge-post {
    display:inline-block;
    background:rgba(255,100,80,0.15);
    color:#ff6450;
    border:1px solid rgba(255,100,80,0.3);
    border-radius:6px;
    padding:2px 10px;
    font-size:0.75rem;
    font-weight:700;
    letter-spacing:.08em;
}

.insight-box {
    background: rgba(255,120,80,0.08);
    border-left: 3px solid #ff7850;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #c8cad8;
}

div[data-testid="stSidebar"] {
    background: rgba(10,14,26,0.95);
    border-right: 1px solid rgba(255,255,255,0.06);
}

div[data-testid="stSidebar"] .stMarkdown p {
    color: #8a8fa8;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)

# ── 데이터 로드 ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("ta_20260601093156.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["날짜"] = df["날짜"].str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    df["계절"] = df["월"].map({12:"겨울",1:"겨울",2:"겨울",
                               3:"봄",4:"봄",5:"봄",
                               6:"여름",7:"여름",8:"여름",
                               9:"가을",10:"가을",11:"가을"})
    df["시대"] = df["연도"].apply(lambda y: "1980년 이전" if y < 1980 else "1980년 이후")
    return df

df = load_data()

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 분석 설정")
    cutoff = st.slider("기준 연도", 1950, 2010, 1980, 5,
                       help="이전/이후를 나누는 기준 연도")
    rolling_n = st.slider("이동평균 연수", 3, 20, 10)
    show_ci = st.checkbox("신뢰구간 표시", True)

    st.markdown("---")
    st.markdown("### 📂 데이터 정보")
    st.markdown(f"- 기간: **{df['연도'].min()}** ~ **{df['연도'].max()}**")
    st.markdown(f"- 관측소: 서울 (108)")
    st.markdown(f"- 총 데이터: **{len(df):,}** 일")
    st.markdown("---")
    st.markdown("""
    <p>데이터 출처: 기상청 기상자료개방포털</p>
    """, unsafe_allow_html=True)

df["시대2"] = df["연도"].apply(lambda y: f"{cutoff}년 이전" if y < cutoff else f"{cutoff}년 이후")

# ── 연간 데이터 집계 ──────────────────────────────────────────
annual = df.groupby("연도").agg(
    평균기온=("평균기온(℃)", "mean"),
    최저기온=("최저기온(℃)", "mean"),
    최고기온=("최고기온(℃)", "mean")
).reset_index()
annual["이동평균"] = annual["평균기온"].rolling(rolling_n, center=True).mean()

pre_df  = annual[annual["연도"] < cutoff]
post_df = annual[annual["연도"] >= cutoff]
pre_mean  = pre_df["평균기온"].mean()
post_mean = post_df["평균기온"].mean()
diff      = post_mean - pre_mean

# 선형회귀 전체
slope, intercept, r, p, se = stats.linregress(annual["연도"], annual["평균기온"])
trend_per_decade = slope * 10

# t-test
t_stat, p_val = stats.ttest_ind(pre_df["평균기온"].dropna(),
                                 post_df["평균기온"].dropna())

# ── 헤더 ────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 16px">
  <div style="font-family:'Space Mono',monospace;font-size:0.7rem;letter-spacing:.2em;
              color:#8a8fa8;text-transform:uppercase;margin-bottom:6px">
    Seoul Climate Analysis · 1907–2026
  </div>
  <h1 style="font-size:2.2rem;font-weight:900;color:#f0f2f8;margin:0;line-height:1.1">
    🌡️ 서울 기온 변화<br><span style="color:#ff7850">1980년대 전후 비교</span>
  </h1>
</div>
""", unsafe_allow_html=True)

# ── 핵심 지표 카드 ────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

def card(col, val, label, color="#ff7850", prefix="", suffix="°C"):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-val" style="color:{color}">{prefix}{val}{suffix}</div>
    </div>
    """, unsafe_allow_html=True)

card(c1, f"{pre_mean:.2f}", f"{cutoff}년 이전 연평균", "#64a0ff")
card(c2, f"{post_mean:.2f}", f"{cutoff}년 이후 연평균", "#ff7850")
card(c3, f"+{diff:.2f}" if diff>0 else f"{diff:.2f}", "기온 상승폭", "#ff4f6e")
card(c4, f"{trend_per_decade:+.3f}", "10년당 추세", "#a78bfa")

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ── 통계적 유의성 ─────────────────────────────────────────────
sig_txt = "✅ 통계적으로 유의미 (p < 0.001)" if p_val < 0.001 else f"p = {p_val:.4f}"
st.markdown(f"""
<div class="insight-box">
  <b>가설 검정 결과 (독립표본 t-검정)</b><br>
  t = {t_stat:.2f}, p-value = {p_val:.2e} → <b>{sig_txt}</b><br>
  {cutoff}년을 기준으로 전후 연평균 기온의 차이는 통계적으로 유의미합니다.
  선형 추세는 10년당 <b>{trend_per_decade:+.3f}°C</b> (R² = {r**2:.3f}).
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── 차트 1: 연간 평균기온 + 추세선 ─────────────────────────────
st.markdown('<div class="section-title">📈 연간 평균기온 추이</div>', unsafe_allow_html=True)

fig1 = go.Figure()

# 전/후 구간 색칠
fig1.add_vrect(x0=annual["연도"].min(), x1=cutoff,
               fillcolor="rgba(100,160,255,0.05)", line_width=0,
               annotation_text=f"{cutoff}년 이전", annotation_position="top left",
               annotation_font_color="#64a0ff", annotation_font_size=11)
fig1.add_vrect(x0=cutoff, x1=annual["연도"].max(),
               fillcolor="rgba(255,100,80,0.05)", line_width=0,
               annotation_text=f"{cutoff}년 이후", annotation_position="top right",
               annotation_font_color="#ff6450", annotation_font_size=11)

# 신뢰구간 (rolling std)
if show_ci:
    roll_std = annual["평균기온"].rolling(rolling_n, center=True).std()
    fig1.add_trace(go.Scatter(
        x=pd.concat([annual["연도"], annual["연도"][::-1]]),
        y=pd.concat([annual["이동평균"]+1.96*roll_std,
                     (annual["이동평균"]-1.96*roll_std)[::-1]]),
        fill="toself", fillcolor="rgba(167,139,250,0.12)",
        line=dict(width=0), name="95% CI", hoverinfo="skip"
    ))

# 연간 산점도
fig1.add_trace(go.Scatter(
    x=annual["연도"], y=annual["평균기온"],
    mode="markers",
    marker=dict(size=4, color=annual["평균기온"],
                colorscale=[[0,"#2563eb"],[0.5,"#7c3aed"],[1,"#dc2626"]],
                showscale=False, opacity=0.7),
    name="연평균기온", hovertemplate="%{x}년: %{y:.2f}°C<extra></extra>"
))

# 이동평균
fig1.add_trace(go.Scatter(
    x=annual["연도"], y=annual["이동평균"],
    mode="lines", line=dict(color="#a78bfa", width=2.5),
    name=f"{rolling_n}년 이동평균"
))

# 선형 추세선
y_trend = intercept + slope * annual["연도"]
fig1.add_trace(go.Scatter(
    x=annual["연도"], y=y_trend,
    mode="lines", line=dict(color="#ff4f6e", width=1.5, dash="dot"),
    name=f"선형 추세 ({slope*10:+.3f}°C/10년)"
))

# 기준 연도 수직선
fig1.add_vline(x=cutoff, line=dict(color="#ffffff", width=1, dash="dash"),
               annotation_text=f"{cutoff}", annotation_font_color="#ffffff",
               annotation_font_size=10)

# 평균선
fig1.add_hline(y=pre_mean, line=dict(color="#64a0ff", width=1, dash="dot"),
               annotation_text=f"이전 평균 {pre_mean:.2f}°C",
               annotation_font_color="#64a0ff", annotation_font_size=10,
               annotation_position="bottom right")
fig1.add_hline(y=post_mean, line=dict(color="#ff6450", width=1, dash="dot"),
               annotation_text=f"이후 평균 {post_mean:.2f}°C",
               annotation_font_color="#ff6450", annotation_font_size=10,
               annotation_position="top right")

fig1.update_layout(
    height=440, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Noto Sans KR", color="#8a8fa8"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)", font=dict(color="#c8cad8")),
    xaxis=dict(showgrid=False, color="#8a8fa8",
               linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
               color="#8a8fa8", linecolor="rgba(255,255,255,0.1)",
               title="평균기온 (°C)"),
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified"
)
st.plotly_chart(fig1, use_container_width=True)

# ── 차트 2 & 3 ────────────────────────────────────────────────
col_left, col_right = st.columns(2)

# ── 박스플롯 ──────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-title">📦 전후 분포 비교</div>',
                unsafe_allow_html=True)

    fig2 = go.Figure()
    groups = [(f"{cutoff}년 이전", pre_df["평균기온"], "#64a0ff"),
              (f"{cutoff}년 이후", post_df["평균기온"], "#ff6450")]
    for name, data, color in groups:
        fig2.add_trace(go.Box(
            y=data, name=name,
            boxpoints="all", jitter=0.3, pointpos=0,
            marker=dict(color=color, size=3, opacity=0.4),
            line=dict(color=color, width=2),
            fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba"),
            hovertemplate=f"<b>{name}</b><br>%{{y:.2f}}°C<extra></extra>"
        ))
    fig2.update_layout(
        height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Noto Sans KR", color="#8a8fa8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   color="#8a8fa8", title="연평균기온 (°C)"),
        xaxis=dict(color="#8a8fa8"),
        showlegend=False, margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── 계절별 비교 ───────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-title">🍂 계절별 기온 변화</div>',
                unsafe_allow_html=True)

    season_order = ["봄", "여름", "가을", "겨울"]
    season_colors_pre  = "#64a0ff"
    season_colors_post = "#ff6450"

    sea = df.groupby(["시대2", "계절"])["평균기온(℃)"].mean().reset_index()
    sea["계절"] = pd.Categorical(sea["계절"], categories=season_order, ordered=True)
    sea = sea.sort_values("계절")

    fig3 = go.Figure()
    for era, color in [(f"{cutoff}년 이전", "#64a0ff"), (f"{cutoff}년 이후", "#ff6450")]:
        sub = sea[sea["시대2"] == era]
        fig3.add_trace(go.Bar(
            x=sub["계절"], y=sub["평균기온(℃)"],
            name=era, marker_color=color,
            opacity=0.85,
            hovertemplate=f"<b>{era}</b> %{{x}}: %{{y:.2f}}°C<extra></extra>"
        ))

    fig3.update_layout(
        height=380, barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Noto Sans KR", color="#8a8fa8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c8cad8")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   color="#8a8fa8", title="평균기온 (°C)"),
        xaxis=dict(color="#8a8fa8"),
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── 차트 4: 월별 히트맵 (10년 단위) ──────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">🗓️ 10년 단위 월별 평균기온 히트맵</div>',
            unsafe_allow_html=True)

df["decade"] = (df["연도"] // 10 * 10).astype(str) + "s"
heat = df.groupby(["decade","월"])["평균기온(℃)"].mean().reset_index()
heat_pivot = heat.pivot(index="decade", columns="월", values="평균기온(℃)")
month_labels = ["1월","2월","3월","4월","5월","6월",
                "7월","8월","9월","10월","11월","12월"]

fig4 = go.Figure(go.Heatmap(
    z=heat_pivot.values,
    x=month_labels,
    y=heat_pivot.index.tolist(),
    colorscale=[[0,"#1e3a8a"],[0.3,"#3b82f6"],[0.5,"#f3f4f6"],
                [0.7,"#f97316"],[1,"#7f1d1d"]],
    hovertemplate="%{y} %{x}: %{z:.2f}°C<extra></extra>",
    text=heat_pivot.values.round(1),
    texttemplate="%{text}",
    textfont=dict(size=9.5, color="rgba(255,255,255,0.7)")
))
fig4.update_layout(
    height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Noto Sans KR", color="#8a8fa8"),
    xaxis=dict(color="#8a8fa8"),
    yaxis=dict(color="#8a8fa8", autorange="reversed"),
    margin=dict(l=10, r=10, t=20, b=10)
)
st.plotly_chart(fig4, use_container_width=True)

# ── 차트 5: 10년 단위 평균 막대 + 누적 편차 ───────────────────
st.markdown("---")
col5a, col5b = st.columns(2)

with col5a:
    st.markdown('<div class="section-title">📊 10년 단위 연평균기온</div>',
                unsafe_allow_html=True)
    dec_annual = annual.copy()
    dec_annual["decade"] = (dec_annual["연도"] // 10 * 10)
    dec_mean = dec_annual.groupby("decade")["평균기온"].mean().reset_index()
    baseline = dec_mean[dec_mean["decade"] < cutoff]["평균기온"].mean()
    dec_mean["anomaly"] = dec_mean["평균기온"] - baseline
    dec_mean["color"] = dec_mean["anomaly"].apply(
        lambda x: "#ff6450" if x >= 0 else "#64a0ff")

    fig5 = go.Figure(go.Bar(
        x=dec_mean["decade"].astype(str)+"s",
        y=dec_mean["평균기온"],
        marker_color=dec_mean["color"],
        text=dec_mean["평균기온"].round(2),
        texttemplate="%{text}°C", textposition="outside",
        textfont=dict(color="#c8cad8", size=10),
        hovertemplate="%{x}: %{y:.2f}°C<extra></extra>"
    ))
    fig5.add_hline(y=baseline, line=dict(color="#ffffff", dash="dot", width=1),
                   annotation_text=f"기준 {baseline:.2f}°C",
                   annotation_font_color="#ffffff", annotation_font_size=9)
    fig5.update_layout(
        height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Noto Sans KR", color="#8a8fa8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   color="#8a8fa8", title="°C", range=[9, 14]),
        xaxis=dict(color="#8a8fa8"),
        margin=dict(l=10, r=10, t=20, b=10), showlegend=False
    )
    st.plotly_chart(fig5, use_container_width=True)

with col5b:
    st.markdown('<div class="section-title">🌊 누적 기온 편차 (기준: 전체 평균)</div>',
                unsafe_allow_html=True)
    overall_mean = annual["평균기온"].mean()
    annual_sorted = annual.sort_values("연도")
    annual_sorted["anomaly"] = annual_sorted["평균기온"] - overall_mean
    annual_sorted["cum_anomaly"] = annual_sorted["anomaly"].cumsum()

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=annual_sorted["연도"], y=annual_sorted["cum_anomaly"],
        mode="lines", fill="tozeroy",
        line=dict(color="#a78bfa", width=2),
        fillcolor="rgba(167,139,250,0.15)",
        hovertemplate="%{x}년 누적편차: %{y:.2f}°C<extra></extra>"
    ))
    fig6.add_vline(x=cutoff, line=dict(color="#ff7850", width=1.5, dash="dash"),
                   annotation_text=str(cutoff), annotation_font_color="#ff7850")
    fig6.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
    fig6.update_layout(
        height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Noto Sans KR", color="#8a8fa8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   color="#8a8fa8", title="누적 편차 (°C)"),
        xaxis=dict(showgrid=False, color="#8a8fa8"),
        margin=dict(l=10, r=10, t=20, b=10), showlegend=False
    )
    st.plotly_chart(fig6, use_container_width=True)

# ── 결론 ─────────────────────────────────────────────────────
st.markdown("---")
pre_std  = pre_df["평균기온"].std()
post_std = post_df["평균기온"].std()
season_diff = (df[df["시대2"]==f"{cutoff}년 이후"].groupby("계절")["평균기온(℃)"].mean() -
               df[df["시대2"]==f"{cutoff}년 이전"].groupby("계절")["평균기온(℃)"].mean())

st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(255,120,80,0.08),rgba(167,139,250,0.08));
            border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:28px 32px">
  <div style="font-size:1.1rem;font-weight:700;color:#f0f2f8;margin-bottom:16px">
    📋 분석 요약 — {cutoff}년 기준
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:0.88rem;color:#c8cad8;line-height:1.8">
    <div>
      <b style="color:#64a0ff">{cutoff}년 이전</b><br>
      연평균기온: <b>{pre_mean:.2f}°C</b> (σ={pre_std:.2f})<br>
      관측 연수: <b>{len(pre_df)}년</b>
    </div>
    <div>
      <b style="color:#ff6450">{cutoff}년 이후</b><br>
      연평균기온: <b>{post_mean:.2f}°C</b> (σ={post_std:.2f})<br>
      관측 연수: <b>{len(post_df)}년</b>
    </div>
  </div>
  <div style="margin-top:16px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.06);
              font-size:0.88rem;color:#c8cad8;line-height:1.9">
    ✅ <b>기온 상승폭</b>: <span style="color:#ff4f6e;font-weight:700">+{diff:.2f}°C</span> 상승<br>
    ✅ <b>선형 추세</b>: 10년당 <span style="color:#a78bfa;font-weight:700">{trend_per_decade:+.3f}°C</span> (R²={r**2:.3f})<br>
    ✅ <b>통계 유의성</b>: t={t_stat:.2f}, p={p_val:.2e} → <span style="color:#4ade80">유의미</span><br>
    ✅ <b>가장 큰 상승 계절</b>: <span style="color:#ff7850;font-weight:700">{season_diff.idxmax()}</span> (+{season_diff.max():.2f}°C)
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
