# Research Summary: Data-Driven Real Estate Investment Decision Model under Uncertainty

## Introduction
Real-estate underwriting is often deterministic despite substantial macro and financing uncertainty. This project develops an integrated quantitative framework combining machine learning valuation, levered cash-flow analytics, and Monte Carlo simulation for risk-aware investment decisions.

## Methodology
- **Data design:** synthetic panel with realistic relationships among property features, location quality, macro state variables, and time dynamics.
- **Predictive modeling:** Linear Regression, Random Forest, and Gradient Boosting with train/validation/test evaluation via R² and RMSE.
- **Financial modeling:** annual levered cash-flow engine with acquisition costs, amortizing debt, NOI, and terminal sale.
- **Risk modeling:** Monte Carlo simulation (12,000 paths) with uncertainty in property growth, rent growth, and borrowing-rate paths via mean-reverting dynamics.
- **Sensitivity/scenarios:** deterministic grid for interest rate, purchase price, and rent; Bear/Base/Bull scenario comparison.

## Results
- Ensemble models outperform linear baselines in out-of-sample valuation quality.
- Return distributions highlight meaningful tail risk even when median outcomes are favorable.
- Interest-rate paths and entry valuation are dominant drivers of NPV and IRR.

## Limitations
- Synthetic data does not capture full market microstructure and transaction frictions.
- Gaussian innovation assumptions may underrepresent heavy-tail events.
- Model excludes taxes, major capex shocks, and active refinancing/optionality.

## Conclusion
The project demonstrates a graduate-level decision framework for real-estate capital allocation: predictive valuation + financing-aware cash-flow modeling + probabilistic downside quantification.
