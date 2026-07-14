# Worst-of Put Pricer

Prices a **worst-of put on 2 stocks** (client sells) by Heston Monte Carlo, calibrated to live market data from Yahoo Finance. Built with Streamlit.

Default trade per spec: 70% strike, 1-year tenor, e.g. TSLA / SPCX.

## Features
- Pick any 2 names from the universe (TSLA, SPCX, NVDA, AAPL, MSFT, AMZN, GOOGL, META, AMD, NFLX)
- **Live market data defaults**: spots, 30d realized vol (→ v₀), 1y realized vol (→ θ), 1y realized correlation — every one overridable with fully adjustable sliders
- **Calendar date picker** for maturity (7 days to 5 years out); tenor computed from the actual date
- Per-asset Heston variance processes, correlated spot shocks, full-truncation Euler, antithetic variates
- Premium as % of notional and USD, 95% CI, risk-neutral probability of exercise
- Vanilla put comparison on each single name → shows the worst-of premium pickup and the client's short-correlation... rather, **long-correlation** position
- Client P&L payoff diagram (short put view) with breakeven, worst-of distribution, sample paths
- 1y rebased price history chart of the two underlyings

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Push to a GitHub repo
2. share.streamlit.io → New app → repo → `app.py` → Deploy

## Validation
- WO put strictly dominates both vanilla puts (12.56% vs 6.49% / 8.85% on a TSLA/SPCX-style setup)
- Price is monotone decreasing in correlation (13.55% at ρ=0.1 → 10.04% at ρ=0.9)
- ρ→1 with identical vols collapses toward the single-name vanilla
- 200,000 paths × 252 steps in ~4s
