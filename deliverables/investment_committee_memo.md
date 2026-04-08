# Investment Committee Memo

## Executive Summary
We evaluated three residential acquisition strategies (Core Stabilized, Value-Add, Opportunistic) using a hybrid real-data underwriting model with walk-forward ML valuation and correlated regime Monte Carlo. Results indicate:
- **Case A (Core)**: resilient downside profile, moderate upside, highest stability.
- **Case B (Value-Add)**: strongest risk-adjusted expected value with manageable tails.
- **Case C (Opportunistic)**: elevated tail risk and frequent hurdle shortfall.

Preliminary recommendation: **Invest in Case B**, **Maintain selective allocation to Case A**, and **Do Not Invest in Case C under current financing regime** unless entry basis is repriced or debt terms improve.

## Methodology
- Data: Metro housing index + macro indicators + transaction-level structural synthesis.
- Valuation: Walk-forward model comparison across Linear, Random Forest, Gradient Boosting.
- Risk engine: 12,000-path correlated Monte Carlo with Bull/Base/Bear regimes.
- Decision metrics: median NPV, median IRR, P(NPV<0), P(IRR<hurdle), VaR(5%), CVaR(5%).

## Key Metrics Interpreted
- **NPV distribution** captures expected equity value creation.
- **IRR hurdle probability** determines mandate compliance.
- **VaR/CVaR** captures left-tail capital impairment risk.

## Risks
1. Macro shocks: rate persistence above expected path compresses levered cash flow.
2. Rent-growth disappointment in weak labor regimes reduces NOI and exit value.
3. Exit-liquidity risk from cap-rate expansion.

## Recommendation and Conditions
### Recommended allocation
- Primary: Case B (Value-Add) with strict execution controls.
- Secondary: Case A as defensive ballast.
- Avoid Case C unless one of the following occurs:
  - purchase discount >10%, or
  - debt spread compression >100 bps, or
  - verified rent uplift with pre-leasing support.

### Monitoring triggers
- Mortgage spread vs policy rate.
- Metro-level rent growth momentum.
- Cap-rate repricing by peer transactions.
