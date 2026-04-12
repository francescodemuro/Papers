# Executive Summary

This project implements an institutional real-estate decision engine that combines walk-forward valuation, financing-aware underwriting, and correlated regime risk simulation.

## What is decision-relevant
- Output is a formal recommendation (Invest / Conditional / Reject), not a model score.
- Tail-risk metrics (P(NPV<0), P(IRR<hurdle), VaR, CVaR) are treated as binding constraints.
- Regime and correlation modeling makes financing and exit risk explicit.

## Key Investment Insights
1. Return ranking alone is insufficient for allocation.
2. Financing volatility can overwhelm operating improvements.
3. Value-add projects require execution-gated capital release.
4. Exit-cap conservatism is critical to preserving equity outcomes.
5. Tail-risk controls improve strategy selection and position sizing.

## Recommendation snapshot
- **Invest:** Value-Add (Case B)
- **Conditional Invest:** Core (Case A)
- **Reject:** Opportunistic (Case C), absent pricing/financing reset

## Why this demonstrates MIT MFin readiness
The project demonstrates quantitative rigor, economic interpretation, and decision-governance discipline—the core skill combination required for graduate-level finance and professional investment roles.
