"""
Worst-of Put Pricer — Heston Monte Carlo
Client sells a worst-of put on 2 stocks. Each stock has its OWN full Heston
parameter set (v0, theta, kappa, xi, rho), entered via editable tables and
seeded from live market data. Built for Streamlit Community Cloud.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Worst-of Put Pricer", page_icon="📉", layout="wide")

TICKER_UNIVERSE = ["TSLA", "SPCX", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "NFLX"]

PARAM_LABELS = [
    "Initial vol √v₀ (%)",
    "Long-run vol √θ (%)",
    "Mean reversion κ",
    "Vol of vol ξ",
    "Spot–vol corr ρ",
]

# ----------------------------------------------------------------------------
# Market data
# ----------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(tickers: tuple):
    """
    Spots, realized vols (30d and long-run) and realized correlation.
    Uses whatever history each ticker has — recent IPOs (e.g. SPCX) just get
    shorter windows. Vols per ticker on its own data; correlation on overlap.
    """
    import yfinance as yf

    px = yf.download(list(tickers), period="1y", auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px[list(tickers)]

    spots, vol_30d, vol_lr, days_used = {}, {}, {}, {}
    for tkr in tickers:
        s = px[tkr].dropna()
        if len(s) < 2:
            raise ValueError(f"No usable price history for {tkr}.")
        rets = np.log(s / s.shift(1)).dropna()
        spots[tkr] = float(s.iloc[-1])
        days_used[tkr] = len(rets)
        vol_lr[tkr] = float(rets.std(ddof=1) * np.sqrt(252)) if len(rets) >= 10 else 0.60
        tail = rets.tail(min(30, len(rets)))
        vol_30d[tkr] = float(tail.std(ddof=1) * np.sqrt(252)) if len(tail) >= 10 else vol_lr[tkr]

    both = np.log(px / px.shift(1)).dropna()
    overlap = len(both)
    corr = float(both.corr().iloc[0, 1]) if overlap >= 20 else 0.50
    meta = {"days_used": days_used, "overlap_days": overlap, "corr_estimated": overlap >= 20}
    return spots, vol_30d, vol_lr, corr, px, meta


# ----------------------------------------------------------------------------
# Pricing engine — worst-of put, fully per-asset Heston, full-truncation Euler
# ----------------------------------------------------------------------------

def price_wo_put(
    v0s, thetas, kappas, xis, rho_svs, corr_assets, K_pct, T, r, q,
    n_paths=50000, n_steps=252, antithetic=True, seed=42, n_sample_paths=40,
):
    """
    Worst-of put, performance based:
        WO(T) = min_i S_i(T) / S_i(0)
        payoff = max(K_pct - WO(T), 0)   (fraction of notional)

    ALL Heston parameters are per-asset arrays:
        v0s, thetas, kappas, xis, rho_svs — one entry per stock.
    """
    rng = np.random.default_rng(seed)
    v0s = np.asarray(v0s, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    kappas = np.asarray(kappas, dtype=float)
    xis = np.asarray(xis, dtype=float)
    rho_svs = np.asarray(rho_svs, dtype=float)
    n_assets = len(v0s)

    dt_ = T / n_steps
    sqrt_dt = np.sqrt(dt_)
    C = np.array([[1.0, corr_assets], [corr_assets, 1.0]]) if n_assets == 2 else np.eye(n_assets)
    L = np.linalg.cholesky(C)

    n_base = n_paths // 2 if antithetic else n_paths
    total = n_base * 2 if antithetic else n_base

    X = np.ones((total, n_assets))          # performance space: X(0) = 1
    v = np.tile(v0s, (total, 1))

    n_show = min(n_sample_paths, total)
    wo_paths = np.empty((n_show, n_steps + 1))
    wo_paths[:, 0] = 1.0

    orth = np.sqrt(1.0 - rho_svs**2)        # per-asset, broadcasts over paths

    for step in range(n_steps):
        Z1 = rng.standard_normal((n_base, n_assets))
        Z2 = rng.standard_normal((n_base, n_assets))
        if antithetic:
            Z1 = np.vstack([Z1, -Z1])
            Z2 = np.vstack([Z2, -Z2])

        eps_S = Z1 @ L.T
        eps_v = rho_svs * eps_S + orth * Z2   # per-asset spot–vol correlation

        v_pos = np.maximum(v, 0.0)
        X *= np.exp((r - q - 0.5 * v_pos) * dt_ + np.sqrt(v_pos) * sqrt_dt * eps_S)
        v += kappas * (thetas - v_pos) * dt_ + xis * np.sqrt(v_pos) * sqrt_dt * eps_v

        wo_paths[:, step + 1] = X[:n_show].min(axis=1)

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
        "wo_T": wo_T, "wo_paths": wo_paths,
        "prob_exercise": float((wo_T < K_pct).mean()),
    }


# ----------------------------------------------------------------------------
# Sidebar — trade terms & simulation settings
# ----------------------------------------------------------------------------

st.sidebar.header("Underlyings")
c1, c2 = st.sidebar.columns(2)
t1 = c1.selectbox("Stock 1", TICKER_UNIVERSE, index=0)
t2 = c2.selectbox("Stock 2", TICKER_UNIVERSE, index=1)
if t1 == t2:
    st.sidebar.error("Pick two different stocks.")
    st.stop()

use_live = st.sidebar.toggle("Use live market data", value=True)

md_ok, px_hist, md_meta = False, None, None
if use_live:
    try:
        with st.spinner(f"Fetching {t1} / {t2} data..."):
            spots_d, vol30_d, vollr_d, corr_mkt, px_hist, md_meta = fetch_market_data((t1, t2))
        md_ok = True
        d1, d2 = md_meta["days_used"][t1], md_meta["days_used"][t2]
        st.sidebar.caption(f"History used: {t1} {d1}d · {t2} {d2}d · overlap {md_meta['overlap_days']}d")
        if not md_meta["corr_estimated"]:
            st.sidebar.warning("Too little overlapping history to estimate correlation — "
                               "defaulting to 0.50. Set it yourself with the slider.")
        elif min(d1, d2) < 60:
            st.sidebar.info("One ticker has a short history (recent IPO) — vol and "
                            "correlation estimates are noisy. Sanity-check the tables.")
    except Exception as e:
        st.sidebar.warning(f"Market data fetch failed ({e}). Enter values manually.")

if md_ok:
    S1_def, S2_def = float(spots_d[t1]), float(spots_d[t2])
    v30_1, v30_2 = float(vol30_d[t1]), float(vol30_d[t2])
    vlr_1, vlr_2 = float(vollr_d[t1]), float(vollr_d[t2])
    corr_def = float(np.clip(corr_mkt, -0.95, 0.99))
else:
    S1_def, S2_def = 250.0, 100.0
    v30_1 = v30_2 = 0.45
    vlr_1 = vlr_2 = 0.50
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

st.sidebar.header("Spots & correlation")
S1 = st.sidebar.number_input(f"{t1} spot", value=round(S1_def, 2), step=1.0)
S2 = st.sidebar.number_input(f"{t2} spot", value=round(S2_def, 2), step=1.0)
corr_assets = st.sidebar.slider("Asset correlation", -0.95, 0.99, round(corr_def, 2), 0.01,
                                help="Defaults to realized correlation over the overlapping history")

st.sidebar.header("Monte Carlo")
n_paths = st.sidebar.select_slider("Paths", options=[10000, 25000, 50000, 100000, 200000], value=50000)
n_steps = st.sidebar.select_slider("Time steps", options=[52, 126, 252, 504], value=252)
antithetic = st.sidebar.checkbox("Antithetic variates", value=True)
seed = st.sidebar.number_input("Seed", value=42, step=1)

# ----------------------------------------------------------------------------
# Main — per-stock Heston tables
# ----------------------------------------------------------------------------

st.title("📉 Worst-of Put Pricer")
st.caption(
    f"Client **sells** a worst-of put on {t1} / {t2} — strike {K_pct:.0%} of spot, "
    f"maturing {maturity:%d %b %Y}. Payoff to the buyer: "
    "max(K% − min(S₁ₜ/S₁₀, S₂ₜ/S₂₀), 0) × notional. Each stock has its own full "
    "Heston parameter set below."
)

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric(f"{t1} spot", f"${S1:,.2f}")
mc2.metric(f"{t2} spot", f"${S2:,.2f}")
mc3.metric("Correlation", f"{corr_assets:.2f}")
mc4.metric("Strike levels", f"${S1 * K_pct:,.2f} / ${S2 * K_pct:,.2f}")


def heston_table(ticker: str, vol0_def: float, vollr_def: float, key: str) -> np.ndarray:
    """Editable 5-row table of Heston parameters for one stock."""
    df = pd.DataFrame({
        "Parameter": PARAM_LABELS,
        "Value": [round(vol0_def * 100, 1), round(vollr_def * 100, 1), 2.0, 0.60, -0.70],
    })
    edited = st.data_editor(
        df, hide_index=True, use_container_width=True, key=key,
        column_config={
            "Parameter": st.column_config.TextColumn(disabled=True),
            "Value": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    return edited["Value"].to_numpy(dtype=float)


st.subheader("Heston parameters — one full set per stock")
col_left, col_right = st.columns(2)
with col_left:
    st.markdown(f"**{t1}**")
    p1 = heston_table(t1, v30_1, vlr_1, key=f"heston_{t1}")
with col_right:
    st.markdown(f"**{t2}**")
    p2 = heston_table(t2, v30_2, vlr_2, key=f"heston_{t2}")
st.caption("Vols in % (seeded from 30d and full-history realized vol). "
           "κ = mean reversion speed, ξ = vol of vol, ρ = spot–vol correlation (clipped to ±0.99).")


def parse_params(p, ticker):
    """Table rows -> (v0, theta, kappa, xi, rho) with validation."""
    vol0, vollr, kappa, xi, rho = p
    problems = []
    if not (1 <= vol0 <= 300):
        problems.append("initial vol should be 1–300%")
    if not (1 <= vollr <= 300):
        problems.append("long-run vol should be 1–300%")
    if kappa <= 0:
        problems.append("κ must be > 0")
    if xi < 0:
        problems.append("ξ must be ≥ 0")
    rho = float(np.clip(rho, -0.99, 0.99))
    if problems:
        st.error(f"{ticker}: " + "; ".join(problems))
        st.stop()
    return (vol0 / 100) ** 2, (vollr / 100) ** 2, float(kappa), float(xi), rho


v0_1, th_1, ka_1, xi_1, rh_1 = parse_params(p1, t1)
v0_2, th_2, ka_2, xi_2, rh_2 = parse_params(p2, t2)

# Feller check per stock
for tkr, ka, th, xi_ in [(t1, ka_1, th_1, xi_1), (t2, ka_2, th_2, xi_2)]:
    if 2 * ka * th <= xi_**2:
        st.info(f"{tkr}: Feller condition violated (2κθ ≤ ξ²) — variance can touch zero. "
                "Full truncation handles it; consider more time steps.")

if px_hist is not None:
    with st.expander("Underlying price history"):
        norm_px = px_hist.apply(lambda s: s / s.loc[s.first_valid_index()] * 100)
        figp = go.Figure()
        for col in norm_px.columns:
            figp.add_trace(go.Scatter(x=norm_px.index, y=norm_px[col], name=col, mode="lines"))
        figp.update_layout(yaxis_title="Rebased to 100", height=320, legend=dict(orientation="h"))
        st.plotly_chart(figp, use_container_width=True)

if st.button("Price it", type="primary"):
    with st.spinner("Simulating..."):
        res = price_wo_put(
            [v0_1, v0_2], [th_1, th_2], [ka_1, ka_2], [xi_1, xi_2], [rh_1, rh_2],
            corr_assets, K_pct, T, r, q,
            n_paths=int(n_paths), n_steps=int(n_steps),
            antithetic=antithetic, seed=int(seed),
        )

    premium_pct = res["price"]
    premium_usd = premium_pct * notional

    p1c, p2c, p3c, p4c = st.columns(4)
    p1c.metric("Premium (client receives)", f"{premium_pct:.2%} of notional")
    p2c.metric("Premium (USD)", f"${premium_usd:,.0f}")
    p3c.metric("95% CI", f"[{res['ci'][0]:.2%}, {res['ci'][1]:.2%}]")
    p4c.metric("P(exercised at T)", f"{res['prob_exercise']:.1%}",
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
        "The worst-of put always costs more than either vanilla — the buyer gets paid on "
        "whichever name does worse. The pickup shrinks as correlation → 1 and grows as it "
        "falls: the client selling this is **long correlation**."
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
            f"client loses; worst case the loss approaches {K_pct - premium_pct:.1%} of notional."
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
    st.info("Edit the two Heston tables above and the trade terms in the sidebar, then hit **Price it**.")

with st.expander("Model notes"):
    st.markdown(
        r"""
**Payoff.** $\max\!\big(K\% - \min_i \tfrac{S_i(T)}{S_i(0)},\, 0\big) \times \text{notional}$,
simulated in performance space so spots only matter for displaying dollar strikes.

**Dynamics.** Each stock follows its **own** Heston process — all five parameters
($v_0$, $\theta$, $\kappa$, $\xi$, $\rho$) are per-stock, set in the tables:

$$dS_i = (r - q)\,S_i\,dt + \sqrt{v_i}\,S_i\,dW_i^S \qquad
dv_i = \kappa_i(\theta_i - v_i)\,dt + \xi_i\sqrt{v_i}\,dW_i^v$$

Stock–stock correlation applies to the price shocks via a Cholesky factor; each
stock's variance shock is then $\rho_i\,\varepsilon_i^S + \sqrt{1-\rho_i^2}\,Z_i$.
Discretisation is full-truncation Euler with antithetic variates.

**Calibration.** Table defaults are seeded from realized data (30d vol → $\sqrt{v_0}$,
full-history vol → $\sqrt{\theta}$); a desk would calibrate each stock's five
parameters to its implied vol surface, and use implied rather than realized correlation.
"""
    )
