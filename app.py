"""
Worst-of Put Pricer — market-implied edition
Client sells a worst-of put on 2 stocks. Headline price uses each stock's
IMPLIED volatility read from its live option chain at the trade's strike and
tenor (skew included), interpolated across expiries in total variance.
A per-stock Heston engine is kept as a stochastic-vol comparison.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Worst-of Put Pricer", page_icon="📉", layout="wide")

TICKER_UNIVERSE = ["TSLA", "SPCX", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "NFLX"]
PARAM_LABELS = ["Initial vol √v₀ (%)", "Long-run vol √θ (%)", "Mean reversion κ", "Vol of vol ξ", "Spot–vol corr ρ"]

# ============================================================================
# MARKET DATA
# ============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_hist(tickers: tuple):
    """Spots, realized vols and realized correlation from price history."""
    import yfinance as yf
    px = yf.download(list(tickers), period="1y", auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px[list(tickers)]
    spots, vol_lr, days_used = {}, {}, {}
    for tkr in tickers:
        s = px[tkr].dropna()
        if len(s) < 2:
            raise ValueError(f"No price history for {tkr}.")
        rets = np.log(s / s.shift(1)).dropna()
        spots[tkr] = float(s.iloc[-1])
        days_used[tkr] = len(rets)
        vol_lr[tkr] = float(rets.std(ddof=1) * np.sqrt(252)) if len(rets) >= 10 else 0.60
    both = np.log(px / px.shift(1)).dropna()
    corr = float(both.corr().iloc[0, 1]) if len(both) >= 20 else 0.50
    return spots, vol_lr, corr, px, {"days_used": days_used, "overlap": len(both),
                                     "corr_est": len(both) >= 20}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_div_yield(ticker: str) -> float:
    """Dividend yield as a decimal; defensive against yfinance format changes."""
    import yfinance as yf
    try:
        dy = yf.Ticker(ticker).info.get("dividendYield") or 0.0
        dy = float(dy)
        if dy > 0.25:          # sometimes returned in percent (e.g. 1.3 meaning 1.3%)
            dy = dy / 100.0
        return float(np.clip(dy, 0.0, 0.10))
    except Exception:
        return 0.0


@st.cache_data(ttl=600, show_spinner=False)
def fetch_riskfree_seed() -> float:
    """Seed the risk-free rate from the 13-week T-bill index; fallback 4%."""
    import yfinance as yf
    try:
        h = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        r = float(h.iloc[-1]) / 100.0
        if 0.0 < r < 0.15:
            return r
    except Exception:
        pass
    return 0.04


def _smile_from_chain(chain, spot):
    """One expiry's smile: OTM options only, junk filtered. -> (moneyness, iv) sorted."""
    puts, calls = chain.puts, chain.calls
    rows = []
    for df, side in ((puts, "put"), (calls, "call")):
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        d = d[(d["bid"] > 0) & (d["ask"] > 0) & d["impliedVolatility"].between(0.01, 4.0)]
        d["m"] = d["strike"] / spot
        d = d[(d["m"] > 0.35) & (d["m"] < 1.8)]
        # OTM side only: puts below spot, calls above (cleaner IVs)
        d = d[d["m"] <= 1.0] if side == "put" else d[d["m"] > 1.0]
        rows.append(d[["m", "impliedVolatility"]])
    if not rows:
        return None
    sm = pd.concat(rows).sort_values("m")
    sm = sm.groupby("m", as_index=False)["impliedVolatility"].mean()
    return sm if len(sm) >= 4 else None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_iv(ticker: str, spot: float, target_T: float):
    """
    Implied vol at ATM and at any moneyness, interpolated to target_T.

    Picks the two listed expiries bracketing the target maturity, builds an
    OTM smile for each, then interpolates linearly in TOTAL VARIANCE across
    time. Returns a dict with iv(m) callable data, diagnostics, and the
    short-dated ATM vol (used to seed Heston v0).
    """
    import yfinance as yf
    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        raise ValueError(f"{ticker}: no listed options")

    today = dt.date.today()
    Ts = np.array([max(( dt.date.fromisoformat(e) - today).days, 1) / 365.0 for e in expiries])
    order = np.argsort(Ts)
    Ts, expiries = Ts[order], [expiries[i] for i in order]

    # bracketing expiries (clamp to ends if target is outside listed range)
    hi_idx = int(np.searchsorted(Ts, target_T))
    lo_idx = max(hi_idx - 1, 0)
    hi_idx = min(hi_idx, len(Ts) - 1)
    extrapolated = target_T > Ts[-1] + 1e-9 or target_T < Ts[0] - 1e-9

    smiles, used = {}, []
    for idx in {lo_idx, hi_idx}:
        sm = _smile_from_chain(tk.option_chain(expiries[idx]), spot)
        if sm is not None:
            smiles[idx] = sm
            used.append((expiries[idx], float(Ts[idx]), len(sm)))
    if not smiles:
        raise ValueError(f"{ticker}: option chains too thin to build a smile")

    def iv_at(idx, m):
        sm = smiles[idx]
        return float(np.interp(m, sm["m"].to_numpy(), sm["impliedVolatility"].to_numpy()))

    def iv(m):
        """IV at moneyness m for target_T via total-variance interpolation."""
        ids = sorted(smiles)
        if len(ids) == 1:
            return iv_at(ids[0], m)
        i0, i1 = ids
        T0, T1 = Ts[i0], Ts[i1]
        w = float(np.clip((target_T - T0) / (T1 - T0), 0.0, 1.0)) if T1 > T0 else 1.0
        tv = (1 - w) * iv_at(i0, m) ** 2 * T0 + w * iv_at(i1, m) ** 2 * T1
        T_eff = (1 - w) * T0 + w * T1
        return float(np.sqrt(max(tv, 1e-8) / max(T_eff, 1e-8)))

    # short-dated ATM vol for Heston v0 seed: nearest expiry >= ~2 weeks with a smile
    v0_seed = None
    for idx in range(len(Ts)):
        if Ts[idx] >= 10 / 365:
            sm = _smile_from_chain(tk.option_chain(expiries[idx]), spot)
            if sm is not None:
                v0_seed = float(np.interp(1.0, sm["m"].to_numpy(), sm["impliedVolatility"].to_numpy()))
                break
    # pack smile points of the longer bracketing expiry for plotting
    plot_idx = max(smiles)
    plot_sm = smiles[plot_idx]
    return {
        "iv_curve_m": plot_sm["m"].tolist(),
        "iv_curve_v": plot_sm["impliedVolatility"].tolist(),
        "plot_expiry": expiries[plot_idx],
        "iv_atm": iv(1.0),
        "iv_fn_points": {int(k): (smiles[k]["m"].tolist(), smiles[k]["impliedVolatility"].tolist(),
                                  float(Ts[k])) for k in smiles},
        "target_T": target_T,
        "used": used,
        "extrapolated": bool(extrapolated),
        "v0_seed": v0_seed,
    }


def iv_from_pack(pack, m):
    """Re-evaluate the total-variance interpolation from cached smile points."""
    pts = pack["iv_fn_points"]
    ids = sorted(pts)
    def one(k):
        ms, vs, _ = pts[k]
        return float(np.interp(m, ms, vs))
    if len(ids) == 1:
        return one(ids[0])
    (k0, k1) = ids
    T0, T1 = pts[k0][2], pts[k1][2]
    tT = pack["target_T"]
    w = float(np.clip((tT - T0) / (T1 - T0), 0.0, 1.0)) if T1 > T0 else 1.0
    tv = (1 - w) * one(k0) ** 2 * T0 + w * one(k1) ** 2 * T1
    T_eff = (1 - w) * T0 + w * T1
    return float(np.sqrt(max(tv, 1e-8) / max(T_eff, 1e-8)))


# ============================================================================
# PRICING ENGINES
# ============================================================================

def price_wo_implied(sig1, sig2, corr, K_pct, T, r, q1, q2,
                     n_paths=500000, seed=42, antithetic=True):
    """
    Headline engine. Terminal-only correlated lognormal draw — EXACT for this
    European payoff (no path dependence → no discretisation error), using each
    stock's implied vol AT THE TRADE'S STRIKE AND TENOR (skew-consistent).
    """
    rng = np.random.default_rng(seed)
    n_base = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal((n_base, 2))
    if antithetic:
        Z = np.vstack([Z, -Z])
    e1 = Z[:, 0]
    e2 = corr * Z[:, 0] + np.sqrt(1 - corr**2) * Z[:, 1]

    X1 = np.exp((r - q1 - 0.5 * sig1**2) * T + sig1 * np.sqrt(T) * e1)
    X2 = np.exp((r - q2 - 0.5 * sig2**2) * T + sig2 * np.sqrt(T) * e2)
    wo = np.minimum(X1, X2)
    disc = np.exp(-r * T)

    def stats(pay):
        if antithetic:
            paired = 0.5 * (pay[:n_base] + pay[n_base:])
            return disc * paired.mean(), disc * paired.std(ddof=1) / np.sqrt(n_base)
        return disc * pay.mean(), disc * pay.std(ddof=1) / np.sqrt(n_base)

    price, se = stats(np.maximum(K_pct - wo, 0.0))
    v1 = stats(np.maximum(K_pct - X1, 0.0))
    v2 = stats(np.maximum(K_pct - X2, 0.0))
    return {"price": price, "se": se, "ci": (price - 1.96 * se, price + 1.96 * se),
            "vanilla": [v1, v2], "wo_T": wo,
            "prob_exercise": float((wo < K_pct).mean())}


def price_wo_heston(v0s, thetas, kappas, xis, rho_svs, corr_assets, K_pct, T, r, qs,
                    n_paths=50000, n_steps=252, antithetic=True, seed=42):
    """Per-asset Heston, full-truncation Euler (comparison engine)."""
    rng = np.random.default_rng(seed)
    v0s, thetas = np.asarray(v0s, float), np.asarray(thetas, float)
    kappas, xis, rho_svs = np.asarray(kappas, float), np.asarray(xis, float), np.asarray(rho_svs, float)
    qs = np.asarray(qs, float)
    n_assets = len(v0s)
    dt_ = T / n_steps
    sqrt_dt = np.sqrt(dt_)
    C = np.array([[1.0, corr_assets], [corr_assets, 1.0]])
    L = np.linalg.cholesky(C)
    n_base = n_paths // 2 if antithetic else n_paths
    total = n_base * 2 if antithetic else n_base
    X = np.ones((total, n_assets))
    v = np.tile(v0s, (total, 1))
    orth = np.sqrt(1.0 - rho_svs**2)
    for _ in range(n_steps):
        Z1 = rng.standard_normal((n_base, n_assets))
        Z2 = rng.standard_normal((n_base, n_assets))
        if antithetic:
            Z1, Z2 = np.vstack([Z1, -Z1]), np.vstack([Z2, -Z2])
        eps_S = Z1 @ L.T
        eps_v = rho_svs * eps_S + orth * Z2
        v_pos = np.maximum(v, 0.0)
        X *= np.exp((r - qs - 0.5 * v_pos) * dt_ + np.sqrt(v_pos) * sqrt_dt * eps_S)
        v += kappas * (thetas - v_pos) * dt_ + xis * np.sqrt(v_pos) * sqrt_dt * eps_v
    wo = X.min(axis=1)
    disc = np.exp(-r * T)
    pay = np.maximum(K_pct - wo, 0.0)
    if antithetic:
        paired = 0.5 * (pay[:n_base] + pay[n_base:])
        price, se = disc * paired.mean(), disc * paired.std(ddof=1) / np.sqrt(n_base)
    else:
        price, se = disc * pay.mean(), disc * pay.std(ddof=1) / np.sqrt(n_base)
    return {"price": price, "se": se}


# ============================================================================
# SIDEBAR — trade terms
# ============================================================================

st.sidebar.header("Underlyings")
c1, c2 = st.sidebar.columns(2)
t1 = c1.selectbox("Stock 1", TICKER_UNIVERSE, index=0)
t2 = c2.selectbox("Stock 2", TICKER_UNIVERSE, index=1)
if t1 == t2:
    st.sidebar.error("Pick two different stocks.")
    st.stop()

st.sidebar.header("Trade")
today = dt.date.today()
maturity = st.sidebar.date_input("Maturity date", value=today + dt.timedelta(days=365),
                                 min_value=today + dt.timedelta(days=7),
                                 max_value=today + dt.timedelta(days=365 * 5))
T = max((maturity - today).days, 1) / 365.0
st.sidebar.caption(f"T = {T:.3f} years ({(maturity - today).days} days)")
K_pct = st.sidebar.slider("Strike (% of spot)", 10, 150, 70, 1) / 100.0
notional = st.sidebar.number_input("Notional (USD)", value=1_000_000, step=100_000)

r_seed = fetch_riskfree_seed()
r = st.sidebar.slider("Risk-free rate (%)", 0.0, 10.0, round(r_seed * 100, 2), 0.05,
                      help="Seeded from the 13-week T-bill (^IRX)") / 100.0

st.sidebar.header("Monte Carlo")
n_paths = st.sidebar.select_slider("Paths", options=[100000, 250000, 500000, 1000000], value=500000)
antithetic = st.sidebar.checkbox("Antithetic variates", value=True)
seed = st.sidebar.number_input("Seed", value=42, step=1)

# ============================================================================
# MAIN
# ============================================================================

st.title("📉 Worst-of Put Pricer — market implied")
st.caption(
    f"Client **sells** a worst-of put on {t1} / {t2}, strike {K_pct:.0%}, maturing "
    f"{maturity:%d %b %Y}. Headline price uses each stock's **implied vol read from its "
    "live option chain at this strike and tenor** — skew included."
)

# ---- fetch everything -------------------------------------------------------
fetch_errors = []
try:
    spots_d, vollr_d, corr_real, px_hist, hmeta = fetch_hist((t1, t2))
    S1, S2 = float(spots_d[t1]), float(spots_d[t2])
except Exception as e:
    st.error(f"Could not fetch price history ({e}). Check tickers / connection.")
    st.stop()

q1, q2 = fetch_div_yield(t1), fetch_div_yield(t2)

iv_packs, iv_strike, iv_source = {}, {}, {}
for tkr, spot in ((t1, S1), (t2, S2)):
    try:
        pack = fetch_iv(tkr, spot, T)
        iv_packs[tkr] = pack
        iv_strike[tkr] = iv_from_pack(pack, K_pct)
        iv_source[tkr] = "implied"
    except Exception as e:
        iv_strike[tkr] = vollr_d[tkr]
        iv_source[tkr] = "realized (fallback)"
        fetch_errors.append(f"{tkr}: {e} — using realized vol {vollr_d[tkr]:.1%} instead.")

for msg in fetch_errors:
    st.warning(msg)

# ---- data summary -----------------------------------------------------------
mcols = st.columns(4)
mcols[0].metric(f"{t1} spot", f"${S1:,.2f}")
mcols[1].metric(f"{t2} spot", f"${S2:,.2f}")
mcols[2].metric(f"{t1} IV @ {K_pct:.0%}K, {T:.2f}y", f"{iv_strike[t1]:.1%}",
                help=f"Source: {iv_source[t1]}")
mcols[3].metric(f"{t2} IV @ {K_pct:.0%}K, {T:.2f}y", f"{iv_strike[t2]:.1%}",
                help=f"Source: {iv_source[t2]}")

for tkr in (t1, t2):
    if tkr in iv_packs:
        used = ", ".join(f"{e} ({n} pts)" for e, _, n in iv_packs[tkr]["used"])
        extra = " ⚠️ target tenor outside listed expiries — flat extrapolation" \
            if iv_packs[tkr]["extrapolated"] else ""
        st.caption(f"{tkr}: smile built from expiries {used}{extra}")

with st.expander("Implied vol smiles (as fetched)"):
    figs = go.Figure()
    for tkr in (t1, t2):
        if tkr in iv_packs:
            p = iv_packs[tkr]
            figs.add_trace(go.Scatter(x=np.array(p["iv_curve_m"]) * 100,
                                      y=np.array(p["iv_curve_v"]) * 100,
                                      mode="lines+markers", name=f"{tkr} ({p['plot_expiry']})"))
    figs.add_vline(x=K_pct * 100, line_dash="dash", line_color="#E45756",
                   annotation_text=f"Strike {K_pct:.0%}")
    figs.add_vline(x=100, line_dash="dot", line_color="#54A24B", annotation_text="ATM")
    figs.update_layout(xaxis_title="Moneyness (% of spot)", yaxis_title="Implied vol (%)",
                       height=380, legend=dict(orientation="h"))
    st.plotly_chart(figs, use_container_width=True)
    st.caption("OTM options only, zero-bid strikes filtered out. The pricer reads the vol "
               "exactly at the strike line, interpolated to your maturity in total variance.")

# ---- pricing inputs (editable, seeded from the surface) ----------------------
st.subheader("Pricing inputs")
ic1, ic2, ic3 = st.columns(3)
sig1 = ic1.number_input(f"{t1} vol used (%)", value=round(iv_strike[t1] * 100, 1),
                        min_value=1.0, max_value=300.0, step=0.5) / 100.0
sig2 = ic2.number_input(f"{t2} vol used (%)", value=round(iv_strike[t2] * 100, 1),
                        min_value=1.0, max_value=300.0, step=0.5) / 100.0
corr_used = ic3.slider("Correlation used", -0.95, 0.99,
                       round(float(np.clip(corr_real, -0.95, 0.99)), 2), 0.01)
st.caption(
    f"Vols seeded from the option surface at the {K_pct:.0%} strike; edit freely. "
    f"Correlation seeded from realized ({corr_real:.2f} over {hmeta['overlap']}d overlap). "
    "Desks typically mark **implied** correlation 5–15 points above realized for worst-of "
    "pricing — nudging the slider up gives a more market-conservative (lower) premium."
)

if st.button("Price it", type="primary"):
    res = price_wo_implied(sig1, sig2, corr_used, K_pct, T, r, q1, q2,
                           n_paths=int(n_paths), seed=int(seed), antithetic=antithetic)
    premium_pct, premium_usd = res["price"], res["price"] * notional

    pc = st.columns(4)
    pc[0].metric("Fair premium (client receives)", f"{premium_pct:.2%} of notional")
    pc[1].metric("Premium (USD)", f"${premium_usd:,.0f}")
    pc[2].metric("95% CI", f"[{res['ci'][0]:.2%}, {res['ci'][1]:.2%}]")
    pc[3].metric("P(exercised at T)", f"{res['prob_exercise']:.1%}",
                 help="Risk-neutral probability the worst performer finishes below strike")
    st.caption("This is **fair value**. A client-facing quote embeds dealer margin — "
               "typically 0.5–3% of notional lower — so expect real term sheets to sit "
               "slightly below this number.")

    st.subheader("Worst-of vs vanilla puts")
    (v1p, _), (v2p, _) = res["vanilla"]
    cc = st.columns(3)
    cc[0].metric(f"Vanilla {K_pct:.0%} put on {t1}", f"{v1p:.2%}")
    cc[1].metric(f"Vanilla {K_pct:.0%} put on {t2}", f"{v2p:.2%}")
    cc[2].metric("Worst-of pickup", f"+{premium_pct - max(v1p, v2p):.2%}",
                 help="Extra premium vs the richer single-name put — the dispersion premium")

    tabs = st.tabs(["Worst-of distribution", "Payoff at maturity (client view)", "Heston comparison"])

    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=res["wo_T"][:200000] * 100, nbinsx=90,
                                   marker_color="#4C78A8", opacity=0.85))
        fig.add_vline(x=K_pct * 100, line_dash="dash", line_color="#E45756",
                      annotation_text=f"Strike {K_pct:.0%}")
        fig.add_vline(x=100, line_dash="dot", line_color="#54A24B", annotation_text="Spot")
        fig.update_layout(title=f"Worst-of performance at maturity — {res['prob_exercise']:.1%} below strike",
                          xaxis_title="min(S₁ₜ/S₁₀, S₂ₜ/S₂₀) (% of spot)", yaxis_title="Paths",
                          showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        wo_grid = np.linspace(0.0, 1.5, 301)
        pnl = premium_pct - np.maximum(K_pct - wo_grid, 0.0)
        breakeven = (K_pct - premium_pct) * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=wo_grid * 100, y=pnl * 100, mode="lines",
                                  line=dict(color="#4C78A8", width=2)))
        fig2.add_hline(y=0, line_color="grey", line_width=1)
        fig2.add_vline(x=K_pct * 100, line_dash="dash", line_color="#E45756",
                       annotation_text=f"Strike {K_pct:.0%}")
        fig2.add_vline(x=breakeven, line_dash="dot", line_color="#F58518",
                       annotation_text=f"Breakeven {breakeven:.1f}%")
        fig2.update_layout(title="Client P&L at maturity (short worst-of put, % of notional)",
                           xaxis_title="Worst-of performance at T (%)",
                           yaxis_title="P&L (% of notional)", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        st.markdown("**Per-stock Heston (stochastic vol view)** — seeded from the implied "
                    "surface: √v₀ from short-dated ATM IV, √θ from ATM IV at your tenor.")
        hcols = st.columns(2)
        tables = {}
        for col, tkr in zip(hcols, (t1, t2)):
            with col:
                st.markdown(f"**{tkr}**")
                if tkr in iv_packs:
                    v0_seed = iv_packs[tkr]["v0_seed"] or iv_packs[tkr]["iv_atm"]
                    th_seed = iv_packs[tkr]["iv_atm"]
                else:
                    v0_seed = th_seed = iv_strike[tkr]
                df = pd.DataFrame({"Parameter": PARAM_LABELS,
                                   "Value": [round(v0_seed * 100, 1), round(th_seed * 100, 1),
                                             2.0, 0.9, -0.6]})
                tables[tkr] = st.data_editor(
                    df, hide_index=True, use_container_width=True, key=f"h_{tkr}",
                    column_config={"Parameter": st.column_config.TextColumn(disabled=True),
                                   "Value": st.column_config.NumberColumn(format="%.2f")})
        pvals = {k: v["Value"].to_numpy(float) for k, v in tables.items()}
        hres = price_wo_heston(
            [(pvals[t1][0] / 100) ** 2, (pvals[t2][0] / 100) ** 2],
            [(pvals[t1][1] / 100) ** 2, (pvals[t2][1] / 100) ** 2],
            [pvals[t1][2], pvals[t2][2]], [pvals[t1][3], pvals[t2][3]],
            [float(np.clip(pvals[t1][4], -0.99, 0.99)), float(np.clip(pvals[t2][4], -0.99, 0.99))],
            corr_used, K_pct, T, r, [q1, q2],
            n_paths=50000, n_steps=252, antithetic=True, seed=int(seed))
        st.metric("Heston price", f"{hres['price']:.2%}",
                  delta=f"{hres['price'] - premium_pct:+.2%} vs implied-vol price")
        st.caption("Uncalibrated beyond the ATM seeds — treat as a sensitivity view, not the "
                   "quote. The headline number already carries the market's skew because it "
                   "reads IV at the actual strike.")
else:
    st.info("Review the vols pulled from the option chains above, adjust if needed, then hit "
            "**Price it**.")

with st.expander("Why this matches the market (and earlier versions didn't)"):
    st.markdown(
        r"""
**Implied, not realized.** Market premiums are set by what options *cost now* — implied
vol — not by trailing realized vol. For single names these can differ by 10–20 vol points,
which at a 70% strike is enormous.

**Skew.** A 70% strike put lives on the steep part of the smile: its IV is well above ATM.
This pricer reads the vol **at the trade's strike**, interpolating across listed strikes,
and across the two bracketing expiries linearly in total variance $\sigma^2 T$.

**Exact simulation.** The payoff depends only on terminal values, so the headline engine
draws terminal prices in one exact lognormal step — no discretisation error, and enough
paths (500k default) to shrink the Monte Carlo interval to a few basis points.

**What's still approximate.** (1) Using each name's strike-IV in a lognormal model is the
standard structurer shortcut — a full local/stochastic-vol calibration would price the
smile *dynamics* too, usually a small effect for a 1y European worst-of. (2) Correlation
is seeded from realized; there is no listed implied-correlation market for stock pairs, so
desks mark it up judgmentally — the slider is there for exactly that. (3) Fair value ≠
client quote: term sheets embed distribution/hedging margin, typically 0.5–3% of notional.
"""
    )
