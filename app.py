"""
Portfolio Optimizer — Modern Portfolio Theory (MPT)
Maximum Sharpe Ratio | Efficient Frontier | Capital Allocation Line

Run:  streamlit run app.py
Requires Python 3.10+
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
# FINANCIAL ENGINEERING CORE
# ═══════════════════════════════════════════════════════════════════════════════

T = 252  # annualization factor (trading days)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_returns(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    """Download adjusted close prices → daily simple returns."""
    raw = yf.download(list(tickers), period=period, auto_adjust=True,
                      progress=False, threads=True)
    prices = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices.dropna(axis=1, how="all").dropna()
    return prices.pct_change().dropna()


def annualize(w: np.ndarray, mu: np.ndarray,
              Sigma: np.ndarray) -> tuple[float, float]:
    """Return (annual_return, annual_volatility) for weight vector w."""
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
    Tangent portfolio: solve
        max  (w'μT − rf) / sqrt(w'ΣTw)
        s.t. Σwi = 1,  wi ≥ 0   (long-only)

    Strategy: minimise −SR via SLSQP with multiple Dirichlet restarts
    to escape local optima.
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
    """Generate n random portfolios (Dirichlet weights) for frontier scatter."""
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
    Compute the true EF by solving:
        min  sqrt(w'ΣTw)
        s.t. w'μT = target,  Σwi=1,  wi≥0
    for a grid of target returns.
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
# VISUALIZATION
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

    # per-portfolio hover: weight breakdown
    hover = [
        "<br>".join(f"<b>{t}</b>: {mc.at[i, t] * 100:.1f}%" for t in tickers)
        for i in mc.index
    ]

    # ── Monte Carlo scatter ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=mc["Volatility"] * 100,
        y=mc["Return"] * 100,
        mode="markers",
        marker=dict(
            size=4,
            color=mc["Sharpe"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Sharpe<br>Ratio", thickness=14, x=1.02),
            opacity=0.55,
        ),
        text=hover,
        hovertemplate=(
            "<b>σ</b>: %{x:.2f}%  <b>E[R]</b>: %{y:.2f}%<br>"
            "<b>SR</b>: %{marker.color:.3f}<br>%{text}<extra>Simulated Portfolio</extra>"
        ),
        name="Simulated Portfolios",
    ))

    # ── Efficient Frontier curve (optional) ─────────────────────────────────
    if ef_vols is not None and len(ef_vols) > 1:
        fig.add_trace(go.Scatter(
            x=ef_vols * 100,
            y=ef_rets * 100,
            mode="lines",
            line=dict(color="#00b4d8", width=3),
            name="Efficient Frontier",
            hoverinfo="skip",
        ))

    # ── Capital Allocation Line ─────────────────────────────────────────────
    slope = (opt_ret - rf) / opt_vol
    x_cal = np.linspace(0.0, opt_vol * 1.7, 150)
    y_cal = rf + slope * x_cal
    fig.add_trace(go.Scatter(
        x=x_cal * 100,
        y=y_cal * 100,
        mode="lines",
        line=dict(color="#ff6b6b", width=2.5, dash="dash"),
        name=f"CAL  (slope = {slope:.3f})",
        hovertemplate="σ: %{x:.2f}%<br>E[R]: %{y:.2f}%<extra>Capital Allocation Line</extra>",
    ))

    # ── Risk-free asset ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0],
        y=[rf * 100],
        mode="markers",
        marker=dict(size=14, symbol="diamond", color="#ff6b6b",
                    line=dict(color="white", width=2)),
        name=f"Risk-Free ({rf * 100:.2f}%)",
        hovertemplate=f"Risk-Free Rate: {rf * 100:.2f}%<extra></extra>",
    ))

    # ── Tangent Portfolio ★ ─────────────────────────────────────────────────
    w_txt = "<br>".join(
        f"<b>{t}</b>: {opt_w[i] * 100:.1f}%" for i, t in enumerate(tickers)
    )
    fig.add_trace(go.Scatter(
        x=[opt_vol * 100],
        y=[opt_ret * 100],
        mode="markers",
        marker=dict(size=22, symbol="star", color="#ffd700",
                    line=dict(color="black", width=1.5)),
        name="Tangent Portfolio ★",
        hovertemplate=(
            "<b>Tangent Portfolio</b><br>"
            f"E[R]: {opt_ret * 100:.2f}%<br>"
            f"σ: {opt_vol * 100:.2f}%<br>"
            f"SR: {opt_sr:.4f}<br><br>"
            f"<b>Weights:</b><br>{w_txt}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Efficient Frontier & Capital Allocation Line",
            font=dict(size=18, family="Arial Black"),
        ),
        xaxis=dict(title="Annualized Volatility (%)", ticksuffix="%",
                   gridcolor="#e8e8e8", zeroline=False),
        yaxis=dict(title="Annualized Expected Return (%)", ticksuffix="%",
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
        title=dict(text="Optimal Portfolio Weights", font=dict(size=16)),
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
        title="Pairwise Correlation Matrix",
        height=max(320, n * 55 + 100),
        margin=dict(l=60, r=40, t=55, b=55),
        paper_bgcolor="white",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Portfolio Optimizer | MPT",
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

st.title("📈 Portfolio Optimizer")
st.markdown("**Modern Portfolio Theory · Maximum Sharpe Ratio · Efficient Frontier**")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    ticker_raw = st.text_area(
        "Tickers (comma or newline-separated)",
        value="AAPL, MSFT, GOOGL, AMZN, JNJ, JPM",
        height=130,
        help="Examples: AAPL, TSLA, BTC-USD, ^GSPC",
    )
    rf = st.slider(
        "Risk-Free Rate (annual)",
        min_value=0.000, max_value=0.100, value=0.045, step=0.005,
        format="%.3f",
        help="US 3-month T-Bill rate or equivalent",
    )
    period = st.selectbox(
        "Historical Period",
        ["1y", "2y", "3y", "5y", "10y"],
        index=3,
    )
    n_sim = st.select_slider(
        "Monte Carlo Simulations",
        options=[1_000, 2_000, 5_000, 10_000],
        value=5_000,
    )
    use_ef = st.toggle(
        "Compute True EF Curve",
        value=False,
        help="Minimises variance for each target return — adds ~15 s",
    )
    go_btn = st.button("🚀  Run Optimization", type="primary", use_container_width=True)
    st.divider()
    st.caption(
        "Data via **yfinance** · Optimizer: **scipy SLSQP** · "
        "Charts: **Plotly**"
    )

# ── Welcome / formula screen ──────────────────────────────────────────────────
if not go_btn:
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.info(
            "Enter tickers in the sidebar and press **Run Optimization** "
            "to compute the Maximum Sharpe Ratio (Tangent) Portfolio."
        )
        st.markdown("#### Optimization Problem")
        st.latex(
            r"\max_{\mathbf{w}} \; SR(\mathbf{w}) = "
            r"\frac{\mathbf{w}^\top \boldsymbol{\mu} \cdot T \;-\; r_f}"
            r"{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \cdot T \cdot \mathbf{w}}}"
        )
        st.latex(
            r"\text{s.t.}\quad \mathbf{1}^\top \mathbf{w} = 1,"
            r"\quad \mathbf{w} \geq \mathbf{0}"
        )
        st.caption(
            "T = 252 (annualization factor) · "
            "**μ** = mean daily returns · "
            "**Σ** = daily covariance matrix"
        )
    with col_r:
        st.markdown("#### How it works")
        st.markdown("""
1. **Fetch** historical prices via yfinance
2. **Compute** μ and Σ from daily returns
3. **Simulate** thousands of random weight vectors → scatter
4. **Optimise** with SLSQP (multiple restarts) → tangent portfolio ★
5. **Draw** the Capital Allocation Line through (0, rf) and ★
        """)
    st.stop()

# ── Parse tickers ─────────────────────────────────────────────────────────────
raw_list = ticker_raw.replace(",", "\n").splitlines()
tickers: list[str] = list(dict.fromkeys(
    t.strip().upper() for t in raw_list if t.strip()
))

if len(tickers) < 2:
    st.error("Please enter **at least 2** tickers.")
    st.stop()

# ── Fetch data ────────────────────────────────────────────────────────────────
with st.spinner(f"Downloading {period} of data for: {', '.join(tickers)} …"):
    try:
        rets = fetch_returns(tuple(tickers), period)
    except Exception as exc:
        st.error(f"Data fetch failed: {exc}")
        st.stop()

valid = list(rets.columns)
dropped = [t for t in tickers if t not in valid]
if dropped:
    st.warning(f"No data retrieved for: **{', '.join(dropped)}**. Skipping.")
tickers = valid

if len(tickers) < 2:
    st.error("Not enough valid tickers. Please check your inputs.")
    st.stop()

mu_vec = rets.mean().values
cov_mat = rets.cov().values

# ── Compute ───────────────────────────────────────────────────────────────────
with st.spinner("Running Monte Carlo simulation …"):
    mc_df = monte_carlo(mu_vec, cov_mat, tickers, n_sim, rf)

with st.spinner("Maximising Sharpe Ratio (SLSQP, multiple restarts) …"):
    opt_w, opt_ret, opt_vol, opt_sr = maximize_sharpe(mu_vec, cov_mat, rf)

ef_vols_arr, ef_rets_arr = None, None
if use_ef:
    with st.spinner("Computing Efficient Frontier curve …"):
        ef_vols_arr, ef_rets_arr = efficient_frontier_curve(mu_vec, cov_mat)

# ── Results ───────────────────────────────────────────────────────────────────
st.success("Optimization complete ✓")
st.subheader("Optimal Portfolio Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected Annual Return",  f"{opt_ret * 100:.2f}%",
          delta=f"+{(opt_ret - rf) * 100:.2f}% vs risk-free")
c2.metric("Annual Volatility (Risk)", f"{opt_vol * 100:.2f}%")
c3.metric("Sharpe Ratio",            f"{opt_sr:.4f}",
          delta="Maximised ↑")
c4.metric("Risk-Free Rate",          f"{rf * 100:.2f}%")

st.divider()

# ── Main charts ───────────────────────────────────────────────────────────────
col_a, col_b = st.columns([3, 2])

with col_a:
    st.plotly_chart(
        fig_frontier(mc_df, opt_w, opt_ret, opt_vol, opt_sr,
                     tickers, rf, ef_vols_arr, ef_rets_arr),
        use_container_width=True,
    )

with col_b:
    st.plotly_chart(fig_pie(opt_w, tickers), use_container_width=True)

    st.markdown("**Allocation Table**")
    alloc_df = (
        pd.DataFrame({"Ticker": tickers, "Allocation": opt_w})
        .assign(Allocation_pct=lambda d: d["Allocation"].map(lambda x: f"{x * 100:.2f}%"))
        .sort_values("Allocation", ascending=False)
        .rename(columns={"Allocation_pct": "Weight"})[["Ticker", "Weight"]]
    )
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

st.divider()

# ── Per-asset statistics ──────────────────────────────────────────────────────
st.subheader("Individual Asset Statistics")

asset_stats = pd.DataFrame({
    "Ticker": tickers,
    "Annual Return":     [f"{mu_vec[i] * T * 100:.2f}%"           for i in range(len(tickers))],
    "Annual Volatility": [f"{np.sqrt(cov_mat[i, i] * T) * 100:.2f}%" for i in range(len(tickers))],
    "Sharpe (solo)":     [
        f"{(mu_vec[i] * T - rf) / (np.sqrt(cov_mat[i, i] * T)):.3f}"
        for i in range(len(tickers))
    ],
    "Portfolio Weight":  [f"{opt_w[i] * 100:.2f}%" for i in range(len(tickers))],
}).sort_values("Portfolio Weight", ascending=False)

st.dataframe(asset_stats, use_container_width=True, hide_index=True)

# ── Correlation heatmap ───────────────────────────────────────────────────────
st.subheader("Correlation Matrix")
st.plotly_chart(fig_heatmap(rets), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ This tool is for **educational purposes only**. "
    "Past performance does not guarantee future results. "
    "Not financial advice."
)
