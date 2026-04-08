# Research Summary: Investment-Grade Real Estate Decision Model

## Introduction
This project addresses real-estate investing as a capital allocation problem where uncertainty in growth, financing, and exit conditions drives outcomes more than point forecasts.

## Methodology
- Built a hybrid dataset using real macro and metro housing index anchors plus structural property-level features.
- Implemented walk-forward validation for ML valuation (Linear, Random Forest, Gradient Boosting).
- Constructed realistic levered cash-flow engine with LTV, interest-only structure, amortization, NOI, and cap-rate-based exit value.
- Ran 12,000-path correlated regime Monte Carlo (Bull/Base/Bear) jointly simulating price growth, rent growth shocks, and interest-rate paths.

## Results
- Time-aware validation improves confidence in model portability across market cycles.
- Risk-adjusted evaluation differentiates strategies more clearly than expected IRR alone.
- Value-Add profile generally dominates on risk-adjusted economics; Opportunistic profile shows materially higher tail impairment risk.

## Limitations
- Bundled data is compact and representative, not full-production breadth.
- Correlation/regime parameters are calibrated for interpretability rather than strict market-implied estimation.

## Conclusion
The framework closes the gap between quant research and actionable investment decisions by combining predictive accuracy, financing realism, and downside-aware portfolio judgment.
