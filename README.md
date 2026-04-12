# Investment-Grade Real Estate Capital Allocation Model

This repository is built as a decision platform, not a classroom demonstration. It integrates quant research and practical underwriting to support capital allocation decisions under macro and financing uncertainty.

## Project Motivation (Linked to Real Allocation Challenges)
In live real-estate decisions, attractive base-case IRRs often fail to survive correlated shocks in financing cost, rent growth, and exit valuation. This project is motivated by that gap: replacing point-estimate underwriting with a risk-adjusted framework that can support investment committee decisions and portfolio sizing.

## Data Credibility and Limits
### Data used
- Macro sample (`data/raw/fred_macro_sample.csv`): policy rate, mortgage rate, inflation, unemployment, GDP growth.
- Housing sample (`data/raw/zillow_zhvi_sample.csv`): metro-level housing and rent index proxies.
- Hybrid transaction layer: structural property-level heterogeneity mapped onto real index anchors.

### Why this is decision-useful
The model preserves macro and market directionality while creating enough micro-level cross-section to train and stress-test underwriting decisions.

### Limitations and decision impact
- Samples are compact and cannot represent full regional heterogeneity.
- Regime and correlation parameters are calibrated rather than fully market-implied.
- Practical implication: model outputs should guide screening and sizing discipline, not replace full due diligence.

## Modeling Stack
1. **Walk-forward ML valuation** for out-of-time robustness.
2. **Levered cash-flow engine** with LTV, IO period, amortization, opex, and cap-rate-based exit.
3. **Correlated regime Monte Carlo** (Bull/Base/Bear) for joint price-rent-rate uncertainty.
4. **Formal decision rule** returning Invest / Conditional Invest / Do Not Invest.

## Formal Decision Framework
The investment decision is based on risk-adjusted thresholds, not headline IRR:
- **Invest**: median NPV ≥ 0, downside probability controlled, and hurdle compliance.
- **Conditional Invest**: economics acceptable but risk limits partially breached.
- **Do Not Invest**: downside or hurdle-failure risk exceeds committee tolerance.

## Key Investment Insights
1. **Risk-adjusted dominance can differ from expected-return ranking.**
   A strategy with slightly lower median IRR may be preferable when P(NPV<0) and CVaR are materially lower, changing allocation toward resilience.
2. **Financing path risk is a first-order driver of equity outcomes.**
   Rate persistence affects debt service and terminal proceeds simultaneously; this supports tighter leverage discipline when rates are volatile.
3. **Value-add economics are highly convex to execution quality.**
   Small rent-uplift misses can move a deal from Invest to Conditional, implying milestone-based capex release is superior to front-loaded deployment.
4. **Exit-cap assumptions dominate late-horizon uncertainty.**
   Cap-rate expansion can erase operating gains, which argues for conservative exit underwriting and tighter acquisition basis.
5. **Tail metrics are operational, not academic.**
   VaR/CVaR directly inform position sizing and concentration limits; ignoring them leads to over-allocation in high-variance strategies.
6. **Regime-aware underwriting changes timing decisions.**
   Deals that pass in base conditions may fail under bear regime probabilities, supporting conditional approvals tied to financing-market triggers.

## Capital Allocation Insight
Without this framework, a committee may approve the highest model IRR profile and unintentionally concentrate left-tail risk. With this framework, decisions are made on risk-adjusted mandate fit, producing different strategy selection, entry pricing discipline, and position sizing.

## Why this project demonstrates readiness for MIT MFin
- Combines predictive modeling with market-aware financial engineering and risk policy design.
- Treats uncertainty as a portfolio decision variable, not a sensitivity afterthought.
- Translates technical outputs into institutional decision language and governance-ready recommendations.
- Demonstrates the mindset expected in graduate-level finance: rigorous modeling, economic interpretation, and action-oriented judgment.

## Repository structure
- `src/real_estate_investment_model.py` — modeling and decision engine.
- `notebooks/Data_Driven_Real_Estate_Investment_Decision_Model.ipynb` — professional quant report notebook.
- `data/` — sample datasets and data documentation.
- `deliverables/` — IC memo, executive summary, slides, CV/SOP lines.
- `REPOSITORY_AUDIT.md` — weakness assessment and upgrade rationale.

## Quick start
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook notebooks/Data_Driven_Real_Estate_Investment_Decision_Model.ipynb
```
