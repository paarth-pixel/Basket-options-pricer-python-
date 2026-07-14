# Heston Basket Option Pricer

Monte Carlo pricer for European basket options where every asset follows Heston stochastic volatility dynamics, with a full asset-asset correlation matrix. Built with Streamlit.

## Features
- Up to 5 assets with editable spots and weights
- Equicorrelation slider or fully custom correlation matrix (auto-repaired to nearest PSD if invalid)
- Heston parameters: v₀, θ, κ, ξ, spot–vol correlation ρ, with live Feller condition check
- Full-truncation Euler scheme, antithetic variates, adjustable paths/steps/seed
- Price, standard error, and 95% confidence interval
- Optional constant-vol GBM benchmark using the expected average variance — shows the stochastic-vol effect directly
- Terminal basket distribution (with % ITM) and sample basket path charts

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (same as previous projects)
1. Push this folder to a new GitHub repo
2. On [share.streamlit.io](https://share.streamlit.io), New app → pick the repo → main file `app.py` → Deploy

## Validation
- Single asset with ξ→0 and v₀=θ reproduces Black–Scholes (9.930 vs 9.925 analytic, within 1 SE)
- Basket put–call parity C − P = B₀ − Ke⁻ʳᵀ holds to MC error
- 5 assets × 50,000 paths × 252 steps prices in ~2.3s
