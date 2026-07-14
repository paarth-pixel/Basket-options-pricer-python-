"""
Heston Basket Option Pricer
Monte Carlo pricing of European basket options where each asset follows
Heston stochastic volatility dynamics, with a full asset-asset correlation
structure. Built for Streamlit Community Cloud.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Heston Basket Option Pricer", page_icon="🧺", layout="wide")

# ----------------------------------------------------------------------------
# Simulation engine
# ----------------------------------------------------------------------------

def nearest_psd(corr: np.ndarray) -> np.ndarray:
    """Clip eigenvalues so a user-entered correlation matrix is usable."""
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, 1e-8, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    return (fixed + fixed.T) / 2


def simulate_heston_basket(
    spots, weights, corr, K, T, r, q,
    v0, kappa, theta, xi, rho_sv,
    option_type="Call", n_paths=20000, n_steps=252,
    antithetic=True, seed=42, n_sample_paths=40,
):
    """
    Full-truncation Euler scheme.

    Each asset i:
        dS_i = (r - q) S_i dt + sqrt(v_i) S_i dW_i^S
        dv_i = kappa (theta - v_i) dt + xi sqrt(v_i) dW_i^v
        corr(dW_i^S, dW_i^v) = rho_sv
        corr(dW_i^S, dW_j^S) = corr[i, j]

    Returns dict with price, std error, CI, terminal baskets, sample paths.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(spots)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    L = np.linalg.cholesky(corr)

    if antithetic:
        n_base = n_paths // 2
    else:
        n_base = n_paths
    total = n_base * 2 if antithetic else n_base

    S = np.tile(np.asarray(spots, dtype=float), (total, 1))
    v = np.full((total, n_assets), v0, dtype=float)

    weights = np.asarray(weights, dtype=float)
    sample_paths = np.empty((min(n_sample_paths, total), n_steps + 1))
    sample_paths[:, 0] = S[: sample_paths.shape[0]] @ weights

    for step in range(n_steps):
        Z1 = rng.standard_normal((n_base, n_assets))
        Z2 = rng.standard_normal((n_base, n_assets))
        if antithetic:
            Z1 = np.vstack([Z1, -Z1])
            Z2 = np.vstack([Z2, -Z2])

        eps_S = Z1 @ L.T                                   # correlated asset shocks
        eps_v = rho_sv * eps_S + np.sqrt(1.0 - rho_sv**2) * Z2

        v_pos = np.maximum(v, 0.0)
        S *= np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * eps_S)
        v += kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqrt_dt * eps_v

        sample_paths[:, step + 1] = S[: sample_paths.shape[0]] @ weights

    basket_T = S @ weights
    if option_type == "Call":
        payoffs = np.maximum(basket_T - K, 0.0)
    else:
        payoffs = np.maximum(K - basket_T, 0.0)

    disc = np.exp(-r * T)
    if antithetic:
        paired = 0.5 * (payoffs[:n_base] + payoffs[n_base:])
        price = disc * paired.mean()
        se = disc * paired.std(ddof=1) / np.sqrt(n_base)
    else:
        price = disc * payoffs.mean()
        se = disc * payoffs.std(ddof=1) / np.sqrt(n_base)

    return {
        "price": price,
        "se": se,
        "ci": (price - 1.96 * se, price + 1.96 * se),
        "basket_T": basket_T,
        "sample_paths": sample_paths,
        "disc_payoff_mean": price,
    }


def simulate_gbm_basket(spots, weights, corr, K, T, r, q, sigma,
                        option_type="Call", n_paths=20000, seed=42):
    """Constant-vol benchmark: one-step exact GBM with the same correlations."""
    rng = np.random.default_rng(seed)
    n_assets = len(spots)
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((n_paths, n_assets)) @ L.T
    ST = np.asarray(spots) * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    basket_T = ST @ np.asarray(weights)
    if option_type == "Call":
        payoffs = np.maximum(basket_T - K, 0.0)
    else:
        payoffs = np.maximum(K - basket_T, 0.0)
    disc = np.exp(-r * T)
    return disc * payoffs.mean(), disc * payoffs.std(ddof=1) / np.sqrt(n_paths)


# ----------------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------------

st.sidebar.header("Basket setup")

n_assets = st.sidebar.slider("Number of assets", 1, 5, 3)

default_spots = [100.0, 95.0, 110.0, 105.0, 90.0][:n_assets]
default_weights = [round(1.0 / n_assets, 4)] * n_assets

asset_df = pd.DataFrame({
    "Asset": [f"Asset {i + 1}" for i in range(n_assets)],
    "Spot": default_spots,
    "Weight": default_weights,
})
asset_df = st.sidebar.data_editor(asset_df, hide_index=True, key="assets",
                                  column_config={"Asset": st.column_config.TextColumn(disabled=True)})

normalize = st.sidebar.checkbox("Normalise weights to sum to 1", value=True)
spots = asset_df["Spot"].to_numpy(dtype=float)
weights = asset_df["Weight"].to_numpy(dtype=float)
if normalize and weights.sum() != 0:
    weights = weights / weights.sum()

st.sidebar.header("Correlation")
rho_assets = st.sidebar.slider("Pairwise asset correlation", -0.30, 0.99, 0.50, 0.01,
                               help="Applied to every asset pair. Use the expander below for a custom matrix.")
corr = np.full((n_assets, n_assets), rho_assets)
np.fill_diagonal(corr, 1.0)

with st.sidebar.expander("Custom correlation matrix"):
    use_custom = st.checkbox("Use custom matrix", value=False)
    corr_df = pd.DataFrame(corr,
                           columns=[f"A{i + 1}" for i in range(n_assets)],
                           index=[f"A{i + 1}" for i in range(n_assets)])
    corr_df = st.data_editor(corr_df, key="corr_matrix")
    if use_custom:
        corr = corr_df.to_numpy(dtype=float)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

# Make sure Cholesky won't blow up on a hand-entered matrix
try:
    np.linalg.cholesky(corr)
except np.linalg.LinAlgError:
    st.sidebar.warning("Correlation matrix isn't positive definite — using nearest valid matrix.")
    corr = nearest_psd(corr)

st.sidebar.header("Option")
option_type = st.sidebar.radio("Type", ["Call", "Put"], horizontal=True)
K = st.sidebar.number_input("Strike K", value=float(round((spots * weights).sum(), 2)), step=1.0)
T = st.sidebar.number_input("Maturity T (years)", value=1.0, min_value=0.01, step=0.25)
r = st.sidebar.number_input("Risk-free rate r", value=0.04, step=0.005, format="%.4f")
q = st.sidebar.number_input("Dividend yield q", value=0.0, step=0.005, format="%.4f")

st.sidebar.header("Heston parameters")
v0 = st.sidebar.number_input("Initial variance v₀", value=0.04, min_value=0.0001, step=0.01, format="%.4f",
                             help="v₀ = 0.04 means initial vol of 20%")
theta = st.sidebar.number_input("Long-run variance θ", value=0.04, min_value=0.0001, step=0.01, format="%.4f")
kappa = st.sidebar.number_input("Mean reversion speed κ", value=2.0, min_value=0.01, step=0.25)
xi = st.sidebar.number_input("Vol of vol ξ", value=0.30, min_value=0.0, step=0.05)
rho_sv = st.sidebar.slider("Spot–vol correlation ρ", -0.99, 0.99, -0.70, 0.01,
                           help="Typically negative for equities (leverage effect)")

st.sidebar.header("Monte Carlo")
n_paths = st.sidebar.select_slider("Paths", options=[5000, 10000, 20000, 50000, 100000], value=20000)
n_steps = st.sidebar.select_slider("Time steps", options=[52, 126, 252, 504], value=252)
antithetic = st.sidebar.checkbox("Antithetic variates", value=True)
seed = st.sidebar.number_input("Seed", value=42, step=1)
show_gbm = st.sidebar.checkbox("Compare vs constant-vol (GBM) benchmark", value=True)

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------

st.title("🧺 Heston Basket Option Pricer")
st.caption(
    "European basket option priced by Monte Carlo. Each asset follows Heston "
    "stochastic volatility with correlated spot shocks across the basket and a "
    "full-truncation Euler discretisation."
)

basket_spot = float((spots * weights).sum())
feller = 2 * kappa * theta - xi**2

info_cols = st.columns(3)
info_cols[0].metric("Basket spot  Σ wᵢSᵢ", f"{basket_spot:,.2f}")
info_cols[1].metric("Moneyness K / basket", f"{K / basket_spot:.2%}" if basket_spot else "—")
info_cols[2].metric("Feller condition  2κθ − ξ²", f"{feller:.4f}",
                    delta="satisfied" if feller > 0 else "violated",
                    delta_color="normal" if feller > 0 else "inverse")
if feller <= 0:
    st.warning("Feller condition violated (2κθ ≤ ξ²): the variance process can hit zero. "
               "Full truncation handles it, but expect a bit more discretisation bias — "
               "consider more time steps.")

if st.button("Price option", type="primary"):
    with st.spinner("Simulating paths..."):
        res = simulate_heston_basket(
            spots, weights, corr, K, T, r, q,
            v0, kappa, theta, xi, rho_sv,
            option_type=option_type, n_paths=int(n_paths), n_steps=int(n_steps),
            antithetic=antithetic, seed=int(seed),
        )

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Basket {option_type.lower()} price", f"{res['price']:.4f}")
    c2.metric("Std error", f"{res['se']:.4f}")
    c3.metric("95% CI", f"[{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]")

    if show_gbm:
        # Expected average variance over [0, T] under CIR dynamics
        avg_var = theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T)
        sigma_eq = float(np.sqrt(avg_var))
        gbm_price, gbm_se = simulate_gbm_basket(
            spots, weights, corr, K, T, r, q, sigma_eq,
            option_type=option_type, n_paths=int(n_paths), seed=int(seed),
        )
        d1, d2, d3 = st.columns(3)
        d1.metric("GBM benchmark price", f"{gbm_price:.4f}",
                  delta=f"{res['price'] - gbm_price:+.4f} Heston − GBM")
        d2.metric("Equivalent constant vol", f"{sigma_eq:.2%}",
                  help="σ = √(expected average variance) = √(θ + (v₀−θ)(1−e^{−κT})/κT)")
        d3.metric("GBM std error", f"{gbm_se:.4f}")
        st.caption(
            "The gap between the two prices is the stochastic-vol effect: with ρ < 0 "
            "Heston fattens the left tail, so OTM puts price above the flat-vol "
            "benchmark and OTM calls below it — the volatility smile."
        )

    tab1, tab2 = st.tabs(["Terminal basket distribution", "Sample basket paths"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=res["basket_T"], nbinsx=80, name="Basket at T",
                                   marker_color="#4C78A8", opacity=0.85))
        fig.add_vline(x=K, line_dash="dash", line_color="#E45756",
                      annotation_text=f"K = {K:g}", annotation_position="top right")
        fig.add_vline(x=basket_spot, line_dash="dot", line_color="#54A24B",
                      annotation_text="Basket spot", annotation_position="top left")
        itm = ((res["basket_T"] > K) if option_type == "Call" else (res["basket_T"] < K)).mean()
        fig.update_layout(title=f"Terminal basket value — {itm:.1%} of paths finish ITM",
                          xaxis_title="Basket value at T", yaxis_title="Paths",
                          showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        t_grid = np.linspace(0, T, res["sample_paths"].shape[1])
        for path in res["sample_paths"]:
            fig2.add_trace(go.Scatter(x=t_grid, y=path, mode="lines",
                                      line=dict(width=1), opacity=0.5,
                                      showlegend=False))
        fig2.add_hline(y=K, line_dash="dash", line_color="#E45756",
                       annotation_text=f"K = {K:g}")
        fig2.update_layout(title=f"{res['sample_paths'].shape[0]} sample basket paths",
                           xaxis_title="Time (years)", yaxis_title="Basket value",
                           height=420)
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Set your parameters in the sidebar, then hit **Price option**.")

with st.expander("Model notes"):
    st.markdown(
        r"""
Each asset follows Heston dynamics under the risk-neutral measure:

$$dS_i = (r - q)\,S_i\,dt + \sqrt{v_i}\,S_i\,dW_i^S \qquad
dv_i = \kappa(\theta - v_i)\,dt + \xi\sqrt{v_i}\,dW_i^v$$

with $\text{corr}(dW_i^S, dW_i^v) = \rho$ (spot–vol) and
$\text{corr}(dW_i^S, dW_j^S)$ from the asset correlation matrix.
Asset shocks are correlated via a Cholesky factor; each variance shock is then
built as $\rho\,\varepsilon_i^S + \sqrt{1-\rho^2}\,Z_i$.

Discretisation is **full-truncation Euler**: the variance is floored at zero
inside both the drift and diffusion terms, which keeps the scheme stable even
when the Feller condition is violated. The payoff is
$\max(\sum_i w_i S_i(T) - K,\,0)$ for a call, discounted at $e^{-rT}$.

There is no closed form for basket options even under Black–Scholes (a sum of
lognormals isn't lognormal), and Heston adds stochastic variance on top — so
Monte Carlo is the natural tool here.
"""
    )
