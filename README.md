# Investment-Grade Real Estate Capital Allocation Model

A professional quantitative research + investment decision repository that combines:
- hybrid real-data underwriting,
- walk-forward machine-learning valuation,
- financing-aware levered cash-flow analytics,
- correlated regime Monte Carlo risk modeling,
- explicit investment recommendations.

## Why this project matters
Most underwriting tools optimize for expected return and underweight tail risk. This repository reframes real-estate selection as a **capital allocation problem under uncertainty**.

## Core outputs
1. **Model reliability:** walk-forward out-of-time ML validation.
2. **Deal economics:** realistic cash-flow model (LTV, interest-only period, amortization, opex, exit cap).
3. **Risk diagnostics:** P(NPV<0), P(IRR<hurdle), VaR, CVaR from correlated simulations.
4. **Decision support:** Invest / Conditional Invest / Do Not Invest recommendations.

## Repository structure
- `src/real_estate_investment_model.py` — full modeling engine.
- `notebooks/Data_Driven_Real_Estate_Investment_Decision_Model.ipynb` — investment-report-style notebook.
- `data/raw/` — bundled macro and housing index samples.
- `data/README.md` — data provenance and limitation notes.
- `deliverables/` — IC memo, executive summary, slide bullets, CV/SOP outputs.
- `REPOSITORY_AUDIT.md` — weaknesses identified and upgrades delivered.

## Quick start
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook notebooks/Data_Driven_Real_Estate_Investment_Decision_Model.ipynb
```

## Capital Allocation Insight
Without this framework, an investor may select the highest-point-estimate IRR strategy and miss structural downside from correlated rent/rate shocks. With this framework, decisions are explicitly tied to mandate-relevant risk constraints (hurdle compliance, NPV impairment probability, left-tail severity), resulting in materially different position sizing and strategy selection.

## Typical recommendation pattern
- **Case A (Core):** defensive, resilient, lower upside.
- **Case B (Value-Add):** best balance of expected value and controlled downside.
- **Case C (Opportunistic):** conditional or reject unless entry basis / financing improves.
