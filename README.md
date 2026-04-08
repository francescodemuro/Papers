# Data-Driven Real Estate Investment Decision Model under Uncertainty

A publication-quality quantitative finance project integrating ML valuation, stochastic risk simulation, and leveraged real-estate cash-flow analysis.

## Problem Statement
This project builds a decision engine for real-estate investments under uncertainty:
- estimate fair value from micro and macro features,
- quantify full risk distributions of returns (not just point forecasts),
- test economic fragility to financing, entry valuation, and rental assumptions.

## Methodology

### 1) Synthetic but economically grounded data
- 30,000 observations (3,000 properties × 10 years).
- Features include property fundamentals, neighborhood indicators, and macro drivers.
- Structural data-generation process includes cyclical and trend behavior.

### 2) Machine learning valuation
- Models: Linear Regression, Random Forest, Gradient Boosting.
- Validation design: train/validation/test split.
- Metrics: R² and RMSE on all splits.
- Model selection by lowest validation RMSE.

### 3) Levered financial model
- Full annual cash-flow stack:
  - acquisition + closing + renovation,
  - leveraged financing with amortization,
  - NOI after vacancy + operating costs,
  - terminal exit net of transaction costs and remaining debt.
- Outputs:
  - NPV,
  - IRR,
  - payback period.

### 4) Monte Carlo risk simulation
- 12,000 paths.
- Stochastic drivers:
  - property appreciation,
  - rental growth,
  - borrowing rate path (mean-reverting process).
- Output distributions for IRR, NPV, sale proceeds, and downside probabilities.

### 5) Sensitivity and scenarios
- Grid sensitivity for interest rate, purchase price, and annual rent.
- Bear/Base/Bull scenario comparison for median return and loss probability.

## Repository Structure
- `src/real_estate_investment_model.py` — modular quant library.
- `notebooks/Data_Driven_Real_Estate_Investment_Decision_Model.ipynb` — complete notebook.
- `research_summary.md` — concise research-style summary.
- `results/figures/` — exported publication-ready figures.

## Reproducibility
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
jupyter notebook notebooks/Data_Driven_Real_Estate_Investment_Decision_Model.ipynb
```

## Key Insights
- Nonlinear ML materially improves valuation accuracy versus linear baselines.
- Strong median returns can coexist with significant downside tail risk.
- Financing conditions and entry valuation are first-order determinants of viability.
