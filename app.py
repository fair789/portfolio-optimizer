"""
ポートフォリオ最適化ツール — 現代ポートフォリオ理論 (MPT)
最大シャープ・レシオ | 効率的フロンティア | 資本配分線

実行方法: streamlit run app.py
Python 3.10+ 必須
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# 金融工学コアロジック
# ═══════════════════════════════════════════════════════════════════════════════

T = 252  # 年率化係数（年間取引日数）


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_returns(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    """修正済み終値をダウンロードし、日次単純リターンを返す。"""
    raw = yf.download(list(tickers), period=period, auto_adjust=True,
                      progress=False, threads=True)
    prices = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices.dropna(axis=1, how="all").dropna()
    return prices.pct_change().dropna()


def annualize(w: np.ndarray, mu: np.ndarray,
              Sigma: np.ndarray) -> tuple[float, float]:
    """ウェイトベクトル w に対する（年率リターン, 年率ボラティリティ）を返す。"""
    ret = float(w @ mu) * T
    vol = float(np.sqrt(w @ (Sigma * T) @ w))
    return ret, vol


def sharpe_ratio(w: np.ndarray, mu: np.ndarray,
                 Sigma: np.ndarray, rf: float) -> float:
    ret, vol = annualize(w, mu, Sigma)
    return (ret - rf) / vol if vol > 1e-10 else 0.0


def maximize_sharpe(
    mu: np.ndarray,
    Sigma: np.ndarray,
    rf: float = 0.045,
    n_restarts: int = 8,
) -> tuple[np.ndarray, float, float, float]:
    """
    接点ポートフォリオ（シャープ・レシオ最大化）を求める。
        max  (w'μT − rf) / sqrt(w'ΣTw)
        s.t. Σwi = 1,  wi ≥ 0（ロングのみ）

    局所解を避けるため、複数のディリクレ乱数初期値から SLSQP で最適化。
    """
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n

    best_sr, best_w = -np.inf, np.ones(n) / n

    for i in range(n_restarts):
        x0 = np.ones(n) / n if i == 0 else np.random.dirichlet(np.ones(n))
        res = minimize(
            lambda w: -sharpe_ratio(w, mu, Sigma, rf),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if res.success and -res.fun > best_sr:
            best_sr = -res.fun
            best_w = res.x.copy()

    best_w = np.maximum(best_w, 0.0)
    best_w /= best_w.sum()
    ret, vol = annualize(best_w, mu, Sigma)
    sr = (ret - rf) / vol
    return best_w, ret, vol, sr


def monte_carlo(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: list[str],
    n: int,
    rf: float,
) -> pd.DataFrame:
    """ランダムなウェイト（ディリクレ分布）で n 個のポートフォリオを生成する。"""
    k = len(mu)
    rows: list[dict] = []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(k))
        ret, vol = annualize(w, mu, Sigma)
        sr = (ret - rf) / vol if vol > 1e-10 else 0.0
        row = {"Return": ret, "Volatility": vol, "Sharpe": sr}
        for j, t in enumerate(tickers):
            row[t] = float(w[j])
        rows.append(row)
    return pd.DataFrame(rows)


def efficient_frontier_curve(
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_points: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """
    各目標リターンに対してボラティリティを最小化することで
    真の効率的フロンティアを計算する。
        min  sqrt(w'ΣTw)
        s.t. w'μT = target,  Σwi=1,  wi≥0
    """
    n = len(mu)
    bounds = [(0.0, 1.0)] * n
    ret_lo = float(mu.min()) * T
    ret_hi = float(mu.max()) * T * 1.05
    vols, rets = [], []

    for target in np.linspace(ret_lo, ret_hi, n_points):
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, tgt=target: annualize(w, mu, Sigma)[0] - tgt},
        ]
        res = minimize(
            lambda w: annualize(w, mu, Sigma)[1],
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if res.success:
            r, v = annualize(res.x, mu, Sigma)
            rets.append(r)
            vols.append(v)

    return np.array(vols), np.array(rets)


# ═══════════════════════════════════════════════════════════════════════════════
# 可視化
# ═══════════════════════════════════════════════════════════════════════════════

_PALETTE = px.colors.qualitative.Plotly


def fig_frontier(
    mc: pd.DataFrame,
    opt_w: np.ndarray,
    opt_ret: float,
    opt_vol: float,
    opt_sr: float,
    tickers: list[str],
    rf: float,
    ef_vols: np.ndarray | None = None,
    ef_rets: np.ndarray | None = None,
) -> go.Figure:
    fig = go.Figure()

    # ホバーテキスト：各ポートフォリオのウェイト内訳
    hover = [
        "<br>".join(f"<b>{t}</b>: {mc.at[i, t] * 100:.1f}%" for t in tickers)
        for i in mc.index
    ]

    # ── モンテカルロ散布図 ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=mc["Volatility"] * 100,
        y=mc["Return"] * 100,
        mode="markers",
        marker=dict(
            size=4,
            color=mc["Sharpe"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="シャープ<br>レシオ", thickness=14, x=1.02),
            opacity=0.55,
        ),
        text=hover,
        hovertemplate=(
            "<b>ボラティリティ</b>: %{x:.2f}%  <b>期待リターン</b>: %{y:.2f}%<br>"
            "<b>シャープ比</b>: %{marker.color:.3f}<br>%{text}<extra>シミュレーション</extra>"
        ),
        name="シミュレーション済みポートフォリオ",
    ))

    # ── 効率的フロンティア曲線（オプション）──────────────────────────────────
    if ef_vols is not None and len(ef_vols) > 1:
        fig.add_trace(go.Scatter(
            x=ef_vols * 100,
            y=ef_rets * 100,
            mode="lines",
            line=dict(color="#00b4d8", width=3),
            name="効率的フロンティア",
            hoverinfo="skip",
        ))

    # ── 資本配分線（CAL）──────────────────────────────────────────────────────
    slope = (opt_ret - rf) / opt_vol
    x_cal = np.linspace(0.0, opt_vol * 1.7, 150)
    y_cal = rf + slope * x_cal
    fig.add_trace(go.Scatter(
        x=x_cal * 100,
        y=y_cal * 100,
        mode="lines",
        line=dict(color="#ff6b6b", width=2.5, dash="dash"),
        name=f"資本配分線  (傾き = {slope:.3f})",
        hovertemplate="ボラティリティ: %{x:.2f}%<br>期待リターン: %{y:.2f}%<extra>資本配分線</extra>",
    ))

    # ── 無リスク資産 ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0],
        y=[rf * 100],
        mode="markers",
        marker=dict(size=14, symbol="diamond", color="#ff6b6b",
                    line=dict(color="white", width=2)),
        name=f"無リスク資産 ({rf * 100:.2f}%)",
        hovertemplate=f"無リスク利子率: {rf * 100:.2f}%<extra></extra>",
    ))

    # ── 接点ポートフォリオ ★ ────────────────────────────────────────────────
    w_txt = "<br>".join(
        f"<b>{t}</b>: {opt_w[i] * 100:.1f}%" for i, t in enumerate(tickers)
    )
    fig.add_trace(go.Scatter(
        x=[opt_vol * 100],
        y=[opt_ret * 100],
        mode="markers",
        marker=dict(size=22, symbol="star", color="#ffd700",
                    line=dict(color="black", width=1.5)),
        name="接点ポートフォリオ ★",
        hovertemplate=(
            "<b>接点ポートフォリオ</b><br>"
            f"期待リターン: {opt_ret * 100:.2f}%<br>"
            f"ボラティリティ: {opt_vol * 100:.2f}%<br>"
            f"シャープ比: {opt_sr:.4f}<br><br>"
            f"<b>投資比率:</b><br>{w_txt}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text="効率的フロンティア & 資本配分線",
            font=dict(size=18, family="Arial Black"),
        ),
        xaxis=dict(title="年率ボラティリティ（リスク）(%)", ticksuffix="%",
                   gridcolor="#e8e8e8", zeroline=False),
        yaxis=dict(title="年率期待リターン (%)", ticksuffix="%",
                   gridcolor="#e8e8e8", zeroline=False),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#ccc",
            borderwidth=1, x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="white",
        height=570,
        hovermode="closest",
        margin=dict(l=65, r=90, t=65, b=60),
    )
    return fig


def fig_pie(opt_w: np.ndarray, tickers: list[str]) -> go.Figure:
    pairs = [(t, w * 100) for t, w in zip(tickers, opt_w) if w * 100 >= 0.5]
    if not pairs:
        pairs = list(zip(tickers, opt_w * 100))
    labels, values = zip(*pairs)

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.42,
        texttemplate="<b>%{label}</b><br>%{value:.1f}%",
        textposition="outside",
        marker=dict(
            colors=_PALETTE[: len(labels)],
            line=dict(color="white", width=2.5),
        ),
        pull=[0.05 if v == max(values) else 0 for v in values],
        hovertemplate="<b>%{label}</b>: %{value:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="最適投資比率", font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="v", x=1.0, y=0.5),
        height=420,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="white",
    )
    return fig


def fig_heatmap(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr().round(3)
    n = len(corr)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text:.2f}",
        textfont=dict(size=11),
        colorbar=dict(title="ρ", thickness=15),
    ))
    fig.update_layout(
        title="銘柄間の相関行列",
        height=max(320, n * 55 + 100),
        margin=dict(l=60, r=40, t=55, b=55),
        paper_bgcolor="white",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Streamlit アプリ
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ポートフォリオ最適化 | MPT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.8rem; }
h1 { font-size: 2.3rem; font-weight: 800; }
div[data-testid="metric-container"] {
    background: #f5f7ff;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border-left: 4px solid #4c6ef5;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 ポートフォリオ最適化ツール")
st.markdown("**現代ポートフォリオ理論 · シャープ・レシオ最大化 · 効率的フロンティア**")
st.divider()

# ── サイドバー ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")

    ticker_raw = st.text_area(
        "銘柄ティッカー（カンマまたは改行で区切る）",
        value="AAPL, MSFT, GOOGL, AMZN, JNJ, JPM",
        height=130,
        help="例: AAPL, TSLA, BTC-USD, ^GSPC（日本株は7203.T など）",
    )
    rf = st.slider(
        "無リスク利子率（年率）",
        min_value=0.000, max_value=0.100, value=0.045, step=0.005,
        format="%.3f",
        help="米国短期国債利回りなどを参考に設定",
    )
    period = st.selectbox(
        "過去データの期間",
        ["1y", "2y", "3y", "5y", "10y"],
        index=3,
        help="期間が長いほど統計的に安定した推計が得られます",
    )
    n_sim = st.select_slider(
        "モンテカルロ・シミュレーション数",
        options=[1_000, 2_000, 5_000, 10_000],
        value=5_000,
    )
    use_ef = st.toggle(
        "効率的フロンティア曲線を計算する",
        value=False,
        help="各目標リターンに対してボラティリティを最小化（約15秒追加）",
    )
    go_btn = st.button("🚀  最適化を実行", type="primary", use_container_width=True)
    st.divider()
    st.caption(
        "データ: **yfinance** · 最適化: **scipy SLSQP** · "
        "グラフ: **Plotly**"
    )

# ── ウェルカム画面 ─────────────────────────────────────────────────────────────
if not go_btn:
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.info(
            "左のサイドバーで銘柄を入力し、**「最適化を実行」** を押してください。"
            "シャープ・レシオを最大化する最適投資比率を計算します。"
        )
        st.markdown("#### 最適化問題（数式）")
        st.latex(
            r"\max_{\mathbf{w}} \; SR(\mathbf{w}) = "
            r"\frac{\mathbf{w}^\top \boldsymbol{\mu} \cdot T \;-\; r_f}"
            r"{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \cdot T \cdot \mathbf{w}}}"
        )
        st.latex(
            r"\text{制約条件:}\quad \mathbf{1}^\top \mathbf{w} = 1,"
            r"\quad \mathbf{w} \geq \mathbf{0}"
        )
        st.caption(
            "T = 252（年間取引日数）· "
            "**μ** = 日次平均リターンベクトル · "
            "**Σ** = 日次共分散行列"
        )
    with col_r:
        st.markdown("#### 計算の流れ")
        st.markdown("""
1. **取得** — yfinance で過去の株価データをダウンロード
2. **推計** — μ（期待リターン）と Σ（共分散行列）を算出
3. **シミュレーション** — ランダムなウェイトで数千のポートフォリオを生成 → 散布図
4. **最適化** — SLSQP（多重再スタート）でシャープ比最大の接点ポートフォリオ ★ を特定
5. **描画** — 無リスク資産と ★ を結ぶ資本配分線（CAL）を描画
        """)
    st.stop()

# ── ティッカー解析 ────────────────────────────────────────────────────────────
raw_list = ticker_raw.replace(",", "\n").splitlines()
tickers: list[str] = list(dict.fromkeys(
    t.strip().upper() for t in raw_list if t.strip()
))

if len(tickers) < 2:
    st.error("銘柄を **2つ以上** 入力してください。")
    st.stop()

# ── データ取得 ────────────────────────────────────────────────────────────────
with st.spinner(f"{period} 分の市場データを取得中: {', '.join(tickers)} …"):
    try:
        rets = fetch_returns(tuple(tickers), period)
    except Exception as exc:
        st.error(f"データ取得に失敗しました: {exc}")
        st.stop()

valid = list(rets.columns)
dropped = [t for t in tickers if t not in valid]
if dropped:
    st.warning(f"データを取得できなかった銘柄をスキップします: **{', '.join(dropped)}**")
tickers = valid

if len(tickers) < 2:
    st.error("有効な銘柄が不足しています。ティッカーをご確認ください。")
    st.stop()

mu_vec = rets.mean().values
cov_mat = rets.cov().values

# ── 計算 ──────────────────────────────────────────────────────────────────────
with st.spinner("モンテカルロ・シミュレーション実行中 …"):
    mc_df = monte_carlo(mu_vec, cov_mat, tickers, n_sim, rf)

with st.spinner("シャープ・レシオを最大化中（SLSQP、多重再スタート）…"):
    opt_w, opt_ret, opt_vol, opt_sr = maximize_sharpe(mu_vec, cov_mat, rf)

ef_vols_arr, ef_rets_arr = None, None
if use_ef:
    with st.spinner("効率的フロンティア曲線を計算中 …"):
        ef_vols_arr, ef_rets_arr = efficient_frontier_curve(mu_vec, cov_mat)

# ── 結果表示 ──────────────────────────────────────────────────────────────────
st.success("最適化が完了しました ✓")
st.subheader("最適ポートフォリオのサマリー")

c1, c2, c3, c4 = st.columns(4)
c1.metric("年率期待リターン", f"{opt_ret * 100:.2f}%",
          delta=f"+{(opt_ret - rf) * 100:.2f}% vs 無リスク利子率")
c2.metric("年率ボラティリティ（リスク）", f"{opt_vol * 100:.2f}%")
c3.metric("シャープ・レシオ", f"{opt_sr:.4f}",
          delta="最大化済み ↑")
c4.metric("無リスク利子率", f"{rf * 100:.2f}%")

st.divider()

# ── メインチャート ────────────────────────────────────────────────────────────
col_a, col_b = st.columns([3, 2])

with col_a:
    st.plotly_chart(
        fig_frontier(mc_df, opt_w, opt_ret, opt_vol, opt_sr,
                     tickers, rf, ef_vols_arr, ef_rets_arr),
        use_container_width=True,
    )

with col_b:
    st.plotly_chart(fig_pie(opt_w, tickers), use_container_width=True)

    st.markdown("**投資比率テーブル**")
    alloc_df = (
        pd.DataFrame({"銘柄": tickers, "比率": opt_w})
        .assign(比率表示=lambda d: d["比率"].map(lambda x: f"{x * 100:.2f}%"))
        .sort_values("比率", ascending=False)
        .rename(columns={"比率表示": "投資比率"})[["銘柄", "投資比率"]]
    )
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

st.divider()

# ── 個別銘柄統計 ──────────────────────────────────────────────────────────────
st.subheader("個別銘柄の統計データ")

asset_stats = pd.DataFrame({
    "銘柄": tickers,
    "年率リターン":     [f"{mu_vec[i] * T * 100:.2f}%"                for i in range(len(tickers))],
    "年率ボラティリティ": [f"{np.sqrt(cov_mat[i, i] * T) * 100:.2f}%"  for i in range(len(tickers))],
    "単独シャープ比":   [
        f"{(mu_vec[i] * T - rf) / (np.sqrt(cov_mat[i, i] * T)):.3f}"
        for i in range(len(tickers))
    ],
    "ポートフォリオ比率": [f"{opt_w[i] * 100:.2f}%" for i in range(len(tickers))],
}).sort_values("ポートフォリオ比率", ascending=False)

st.dataframe(asset_stats, use_container_width=True, hide_index=True)

# ── 相関行列 ──────────────────────────────────────────────────────────────────
st.subheader("銘柄間の相関行列")
st.plotly_chart(fig_heatmap(rets), use_container_width=True)

# ── フッター ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ 本ツールは **教育・学習目的** のみを想定しています。"
    "過去のデータは将来の結果を保証するものではありません。"
    "投資判断は自己責任でお願いします。"
)
