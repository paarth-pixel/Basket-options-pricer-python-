"""
Worst-of Put Pricer — Heston Monte Carlo
Client sells a worst-of put on 2 stocks. Market data (spots, realized vols,
realized correlation) pulled from Yahoo Finance as defaults; every parameter
fully adjustable. Built for Streamlit Community Cloud.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Worst-of Put Pricer", page_icon="📉", layout="wide")

TICKER_UNIVERSE = ["TSLA", "SPCX", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "NFLX"]

# ----------------------------------------------------------------------------
# Market data
# ----------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(tickers: tuple):
    """Spots, realized vols (30d and 1y), and 1y realized correlation."""
    import yfinance as yf

    px = yf.download(list(tickers), period="1y", auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px[list(tickers)].dropna()
    if len(px) < 40:
        raise ValueError("Not enough price history returned.")

    rets = np.log(px / px.shift(1)).dropna()
    spots = px.iloc[-1].to_dict()
    vol_1y = (rets.std() * np.sqrt(252)).to_dict()
    vol_30d = (rets.tail(30).std() * np.sqrt(252)).to_dict()
    corr = float(rets.corr().iloc[0, 1]) if len(tickers) == 2 else 1.0
    return spots, vol_30d, vol_1y, corr, px


# ----------------------------------------------------------------------------
# Pricing engine — worst-of put under 2-asset Heston, full-truncation Euler
# ----------------------------------------------------------------------------

def price_wo_put(
    spots, v0s, thetas, corr_assets, K_pct, T, r, q,
    kappa, xi, rho_sv,
    n_paths=50000, n_steps=252, antithetic=True, seed=42, n_sample_paths=40,
):
    """
    Worst-of put, performance based:
        WO(T) = min_i S_i(T) / S_i(0)
        payoff = max(K_pct - WO(T), 0)   (as a fraction of notional)

    Per-asset v0 and theta; kappa, xi, rho_sv shared.
    Also prices vanilla puts on each single name (same strike %, same paths)
    for comparison, and returns everything needed for the charts.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(spots)
    dt_ = T / n_steps
    sqrt_dt = np.sqrt(dt_)

    C = np.array([[1.0, corr_assets], [corr_assets, 1.0]]) if n_assets == 2 else np.eye(n_assets)
    L = np.linalg.cholesky(C)

    n_base = n_paths // 2 if antithetic else n_paths
    total = n_base * 2 if antithetic else n_base

    # Work in performance space: X_i(0) = 1
    X = np.ones((total, n_assets))
    v = np.tile(np.asarray(v0s, dtype=float), (total, 1))
    thetas = np.asarray(thetas, dtype=float)

    n_show = min(n_sample_paths, total)
    wo_paths = np.empty((n_show, n_steps + 1))
    wo_paths[:, 0] = 1.0

    for _ in range(n_steps):
        Z1 = rng.standard_normal((n_base, n_assets))
        Z2 = rng.standard_normal((n_base, n_assets))
        if antithetic:
            Z1 = np.vstack([Z1, -Z1])
            Z2 = np.vstack([Z2, -Z2])

        eps_S = Z1 @ L.T
        eps_v = rho_sv * eps_S + np.sqrt(1.0 - rho_sv**2) * Z2

        v_pos = np.maximum(v, 0.0)
        X *= np.exp((r - q - 0.5 * v_pos) * dt_ + np.sqrt(v_pos) * sqrt_dt * eps_S)
        v += kappa * (thetas - v_pos) * dt_ + xi * np.sqrt(v_pos) * sqrt_dt * eps_v

        wo_paths[:, _ + 1] = X[:n_show].min(axis=1)

    wo_T = X.min(axis=1)
    disc = np.exp(-r * T)

    def mc_stats(payoffs):
        if antithetic:
            paired = 0.5 * (payoffs[:n_base] + payoffs[n_base:])
            return disc * paired.mean(), disc * paired.std(ddof=1) / np.sqrt(n_base)
        return disc * payoffs.mean(), disc * payoffs.std(ddof=1) / np.sqrt(n_base)

    wo_price, wo_se = mc_stats(np.maximum(K_pct - wo_T, 0.0))
    vanilla = [mc_stats(np.maximum(K_pct - X[:, i], 0.0)) for i in range(n_assets)]

    return {
        "price": wo_price, "se": wo_se,
        "ci": (wo_price - 1.96 * wo_se, wo_price + 1.96 * wo_se),
        "vanilla": vanilla,
        "wo_T": wo_T, "perf_T": X, "wo_paths": wo_paths,
        "prob_exercise": float((wo_T < K_pct).mean()),
    }


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.header("Underlyings")
c1, c2 = st.sidebar.columns(2)
t1 = c1.selectbox("Stock 1", TICKER_UNIVERSE, index=0)
t2 = c2.selectbox("Stock 2", TICKER_UNIVERSE, index=1)

if t1 == t2:
    st.sidebar.error("Pick two different stocks.")
    st.stop()

use_live = st.sidebar.toggle("Use live market data", value=True)

md_ok, px_hist = False, None
if use_live:
    try:
        with st.spinner(f"Fetching {t1} / {t2} data..."):
            spots_d, vol30_d, vol1y_d, corr_mkt, px_hist = fetch_market_data((t1, t2))
        md_ok = True
    except Exception as e:
        st.sidebar.warning(f"Market data fetch failed ({e}). Enter values manually.")

if md_ok:
    S1_def, S2_def = float(spots_d[t1]), float(spots_d[t2])
    vol1_def, vol2_def = float(vol30_d[t1]), float(vol30_d[t2])
    lt_vol1_def, lt_vol2_def = float(vol1y_d[t1]), float(vol1y_d[t2])
    corr_def = float(np.clip(corr_mkt, -0.95, 0.99))
else:
    S1_def, S2_def = 250.0, 100.0
    vol1_def = vol2_def = 0.45
    lt_vol1_def = lt_vol2_def = 0.50
    corr_def = 0.50

st.sidebar.header("Trade")
today = dt.date.today()
maturity = st.sidebar.date_input(
    "Maturity date", value=today + dt.timedelta(days=365),
    min_value=today + dt.timedelta(days=7), max_value=today + dt.timedelta(days=365 * 5),
)
T = max((maturity - today).days, 1) / 365.0
st.sidebar.caption(f"T = {T:.3f} years ({(maturity - today).days} days)")

K_pct = st.sidebar.slider("Strike (% of spot)", 10, 150, 70, 1) / 100.0
notional = st.sidebar.number_input("Notional (USD)", value=1_000_000, step=100_000)
r = st.sidebar.slider("Risk-free rate (%)", 0.0, 10.0, 4.0, 0.05) / 100.0
q = st.sidebar.slider("Dividend yield (%)", 0.0, 8.0, 0.0, 0.05) / 100.0

st.sidebar.header("Model — fully adjustable")
S1 = st.sidebar.number_input(f"{t1} spot", value=round(S1_def, 2), step=1.0)
S2 = st.sidebar.number_input(f"{t2} spot", value=round(S2_def, 2), step=1.0)
corr_assets = st.sidebar.slider("Asset correlation", -0.95, 0.99, round(corr_def, 2), 0.01,
                                help="Defaults to 1y realized correlation of daily log returns")
vol1 = st.sidebar.slider(f"{t1} initial vol √v₀ (%)", 5, 200, int(round(vol1_def * 100)), 1,
                         help="Defaults to 30d realized vol") / 100.0
vol2 = st.sidebar.slider(f"{t2} initial vol √v₀ (%)", 5, 200, int(round(vol2_def * 100)), 1) / 100.0
lt_vol1 = st.sidebar.slider(f"{t1} long-run vol √θ (%)", 5, 200, int(round(lt_vol1_def * 100)), 1,
                            help="Defaults to 1y realized vol") / 100.0
lt_vol2 = st.sidebar.slider(f"{t2} long-run vol √θ (%)", 5, 200, int(round(lt_vol2_def * 100)), 1) / 100.0
kappa = st.sidebar.slider("Mean reversion κ", 0.1, 10.0, 2.0, 0.1)
xi = st.sidebar.slider("Vol of vol ξ", 0.0, 2.0, 0.6, 0.05)
rho_sv = st.sidebar.slider("Spot–vol correlation ρ", -0.99, 0.5, -0.70, 0.01)

st.sidebar.header("Monte Carlo")
n_paths = st.sidebar.select_slider("Paths", options=[10000, 25000, 50000, 100000, 200000], value=50000)
n_steps = st.sidebar.select_slider("Time steps", options=[52, 126, 252, 504], value=252)
antithetic = st.sidebar.checkbox("Antithetic variates", value=True)
seed = st.sidebar.number_input("Seed", value=42, step=1)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

st.title("📉 Worst-of Put Pricer")
st.caption(
    f"Client **sells** a worst-of put on {t1} / {t2} — strike {K_pct:.0%} of spot, "
    f"maturing {maturity:%d %b %Y}. Payoff to the buyer: "
    "max(K% − min(S₁ₜ/S₁₀, S₂ₜ/S₂₀), 0) × notional. Priced by Heston Monte Carlo."
)

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric(f"{t1} spot", f"${S1:,.2f}")
mc2.metric(f"{t2} spot", f"${S2:,.2f}")
mc3.metric("Realized corr (1y)" if md_ok else "Correlation", f"{corr_assets:.2f}")
mc4.metric("Strike levels", f"${S1 * K_pct:,.2f} / ${S2 * K_pct:,.2f}")

if px_hist is not None:
    with st.expander("Underlying price history (1y)"):
        norm_px = px_hist / px_hist.iloc[0] * 100
        figp = go.Figure()
        for col in norm_px.columns:
            figp.add_trace(go.Scatter(x=norm_px.index, y=norm_px[col], name=col, mode="lines"))
        figp.update_layout(yaxis_title="Rebased to 100", height=320, legend=dict(orientation="h"))
        st.plotly_chart(figp, use_container_width=True)

if st.button("Price it", type="primary"):
    with st.spinner("Simulating..."):
        res = price_wo_put(
            [S1, S2], [vol1**2, vol2**2], [lt_vol1**2, lt_vol2**2],
            corr_assets, K_pct, T, r, q, kappa, xi, rho_sv,
            n_paths=int(n_paths), n_steps=int(n_steps),
            antithetic=antithetic, seed=int(seed),
        )

    premium_pct = res["price"]
    premium_usd = premium_pct * notional

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Premium (client receives)", f"{premium_pct:.2%} of notional")
    p2.metric("Premium (USD)", f"${premium_usd:,.0f}")
    p3.metric("95% CI", f"[{res['ci'][0]:.2%}, {res['ci'][1]:.2%}]")
    p4.metric("P(exercised at T)", f"{res['prob_exercise']:.1%}",
              help="Probability the worst performer finishes below the strike (risk-neutral)")

    st.subheader("Worst-of vs vanilla puts — the correlation trade")
    v1_price, _ = res["vanilla"][0]
    v2_price, _ = res["vanilla"][1]
    cA, cB, cC = st.columns(3)
    cA.metric(f"Vanilla {K_pct:.0%} put on {t1}", f"{v1_price:.2%}")
    cB.metric(f"Vanilla {K_pct:.0%} put on {t2}", f"{v2_price:.2%}")
    cC.metric("Worst-of premium pickup", f"+{premium_pct - max(v1_price, v2_price):.2%}",
              help="Extra premium vs the richer single-name put")
    st.caption(
        "The worst-of put always costs more than either vanilla put — the buyer gets paid "
        "on whichever name does worse. The pickup shrinks as correlation → 1 (the basket "
        "behaves like one stock) and grows as correlation falls. The client selling this "
        "is effectively **long correlation**: if realized correlation comes in lower than "
        "priced, the structure was sold too cheap."
    )

    tab1, tab2, tab3 = st.tabs(["Worst-of distribution", "Payoff at maturity (client view)", "Sample paths"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=res["wo_T"] * 100, nbinsx=80,
                                   marker_color="#4C78A8", opacity=0.85))
        fig.add_vline(x=K_pct * 100, line_dash="dash", line_color="#E45756",
                      annotation_text=f"Strike {K_pct:.0%}")
        fig.add_vline(x=100, line_dash="dot", line_color="#54A24B", annotation_text="Spot")
        fig.update_layout(title=f"Worst-of performance at maturity — {res['prob_exercise']:.1%} of paths below strike",
                          xaxis_title="min(S₁ₜ/S₁₀, S₂ₜ/S₂₀)  (% of spot)",
                          yaxis_title="Paths", showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        wo_grid = np.linspace(0.0, 1.5, 301)
        client_pnl = premium_pct - np.maximum(K_pct - wo_grid, 0.0)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=wo_grid * 100, y=client_pnl * 100, mode="lines",
                                  line=dict(color="#4C78A8", width=2), name="Client P&L"))
        fig2.add_hline(y=0, line_color="grey", line_width=1)
        fig2.add_vline(x=K_pct * 100, line_dash="dash", line_color="#E45756",
                       annotation_text=f"Strike {K_pct:.0%}")
        breakeven = (K_pct - premium_pct) * 100
        fig2.add_vline(x=breakeven, line_dash="dot", line_color="#F58518",
                       annotation_text=f"Breakeven {breakeven:.1f}%")
        fig2.update_layout(title="Client P&L at maturity (short worst-of put, % of notional)",
                           xaxis_title="Worst-of performance at T (% of spot)",
                           yaxis_title="P&L (% of notional)", height=420)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            f"Max gain = premium ({premium_pct:.2%}). Below the {breakeven:.1f}% breakeven the "
            "client starts losing; worst case (both stocks → 0) the loss approaches "
            f"{K_pct - premium_pct:.1%} of notional."
        )

    with tab3:
        fig3 = go.Figure()
        t_grid = np.linspace(0, T, res["wo_paths"].shape[1])
        for path in res["wo_paths"]:
            fig3.add_trace(go.Scatter(x=t_grid, y=path * 100, mode="lines",
                                      line=dict(width=1), opacity=0.5, showlegend=False))
        fig3.add_hline(y=K_pct * 100, line_dash="dash", line_color="#E45756",
                       annotation_text=f"Strike {K_pct:.0%}")
        fig3.update_layout(title=f"{res['wo_paths'].shape[0]} sample worst-of paths",
                           xaxis_title="Time (years)", yaxis_title="Worst-of performance (%)",
                           height=420)
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Adjust anything in the sidebar (all sliders live, maturity via the calendar), then hit **Price it**.")

with st.expander("Model notes"):
    st.markdown(
        r"""
**Payoff.** Performance-based worst-of put:
$\text{payoff} = \max\!\big(K\% - \min_i \tfrac{S_i(T)}{S_i(0)},\, 0\big) \times \text{notional}$.
Working in performance space means spots only matter for displaying dollar strike levels.

**Dynamics.** Each stock follows Heston under the risk-neutral measure with its own
$v_0$ (seeded from 30-day realized vol) and $\theta$ (seeded from 1-year realized vol);
$\kappa$, $\xi$ and the spot–vol correlation $\rho$ are shared. Stock–stock correlation
defaults to the 1-year realized correlation of daily log returns. Discretisation is
full-truncation Euler with antithetic variates.

**Why the client is long correlation.** The buyer of a worst-of put owns the *dispersion*
between the two names: the further apart they can drift, the worse the worst performer.
High correlation kills dispersion, so the option cheapens. The seller therefore profits
if correlation realizes higher than the level it was priced at — a key risk to flag,
because equity correlations tend to spike exactly in the sell-offs where the put pays out.

**Caveats.** Realized vol/correlation are backward-looking stand-ins for implied levels a
desk would actually calibrate to; there's no smile calibration here. Good enough to
understand the product's mechanics and sensitivities, not to quote a live market.
"""
    )
