# Research Summary: Investment-Grade Real Estate Decision Engine

## Introduction
This project treats real-estate investing as a risk-adjusted capital allocation problem, motivated by practical underwriting failures where base-case returns obscured downside impairment.

## Methodology
- Hybrid data design: macro and metro housing index anchors with structured property-level heterogeneity.
- Walk-forward ML validation to assess out-of-time valuation robustness.
- Financing-aware levered cash-flow model (LTV, IO period, amortization, NOI, exit-cap terminal valuation).
- Correlated Bull/Base/Bear Monte Carlo jointly simulating price growth, rent shocks, and financing rates.
- Policy-based decision rule producing Invest / Conditional / Reject outputs.

## Decision-Relevant Findings
- Risk-adjusted strategy ranking diverges from expected IRR ranking.
- Financing path volatility is a major determinant of equity downside.
- Exit-cap stress materially affects terminal value and should constrain acquisition basis.
- Tail metrics (VaR/CVaR) are operational inputs for position sizing and concentration limits.

## Limitations
- Compact bundled data supports reproducibility but not full production breadth.
- Regime probabilities and correlation parameters are calibrated for disciplined screening rather than market-implied precision.

## Conclusion
The framework bridges quant research and investment governance by coupling predictive modeling with decision policy, downside controls, and capital-allocation judgment.

## Why this demonstrates readiness for MIT MFin
It demonstrates the ability to integrate statistics, asset-pricing intuition, and institutional decision logic into a coherent investment process—precisely the mindset expected of an incoming MFin candidate.
