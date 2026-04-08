# Executive Summary (1 Page)

This project implements an investment-grade real-estate decision model integrating predictive valuation, financing-aware cash-flow underwriting, and correlated regime Monte Carlo risk analytics.

## What changed vs a typical academic project
- Introduced hybrid real-data architecture (macro + housing indices).
- Replaced static holdout testing with walk-forward validation.
- Modeled cross-factor dependence between price growth, rent growth, and interest rates.
- Converted outputs into direct capital allocation recommendations.

## Decision outcome
- **Invest**: Value-Add strategy (Case B), subject to execution controls.
- **Conditional Invest**: Core strategy (Case A), depending on portfolio diversification need and pricing discipline.
- **Do Not Invest**: Opportunistic strategy (Case C) under current base financing assumptions.

## Why this matters
Without this framework, the committee could over-allocate to high-upside narratives while underpricing left-tail impairment risk. With it, decisions are based on full distributions and mandate alignment, not point estimates.
