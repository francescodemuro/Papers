# Research Summary: Data-Driven Real Estate Investment Decision Model under Uncertainty

## Introduction
Real estate allocation decisions require integrating predictive valuation, financing structure, and uncertainty quantification. Traditional deterministic underwriting often obscures downside risk and parameter sensitivity. This project develops a coherent quantitative framework that merges machine learning, cash-flow valuation, and stochastic simulation to support institutional-quality investment decisions.

## Methodology

### Data and feature design
A synthetic but economically grounded panel dataset was generated to emulate transaction-level real-estate data over time. The data-generating process includes:
- property-level hedonic characteristics,
- location quality proxies,
- macroeconomic state variables,
- trend and cyclical dynamics.

Feature engineering includes non-linear and interaction terms to capture realistic pricing behavior and improve model expressiveness.

### Predictive modeling
Three supervised regression models were benchmarked: Linear Regression, Random Forest, and Gradient Boosting. Models were trained and validated using out-of-sample R² and RMSE to identify a robust pricing engine.

### Financial valuation
A levered cash-flow model was implemented with acquisition costs, mortgage amortization, net operating income, and terminal sale proceeds. Investment performance was assessed via NPV, IRR, and payback period.

### Risk and uncertainty
Monte Carlo simulation (12,000 paths) was used to model uncertainty in annual property appreciation and rental growth. For each path, full investment cash flows were recomputed and mapped to IRR/NPV distributions.

### Sensitivity and scenario analysis
A grid-based sensitivity framework measured valuation elasticity to:
- interest rates,
- purchase price,
- rental levels.

Bear/Base/Bull scenarios were then used to compare median returns and downside probabilities in distinct market regimes.

## Results
- Nonlinear ensemble methods outperform linear baselines in out-of-sample pricing accuracy.
- Stochastic return distributions reveal that attractive expected returns can coexist with nontrivial downside tail risk.
- Financing cost and entry price are dominant drivers of NPV/IRR outcomes.
- Scenario analysis provides clear decision boundaries under regime uncertainty.

## Limitations
- Synthetic data, while structurally realistic, cannot fully replicate local market microstructure.
- Growth-rate innovations are modeled with simple Gaussian assumptions and may understate fat tails.
- The model abstracts from taxes, capex shocks, and dynamic refinancing options.

## Conclusion
The project demonstrates a graduate-level, integrated quant-finance workflow for real-estate investment analytics. By combining ML prediction, levered cash-flow valuation, Monte Carlo risk simulation, and sensitivity diagnostics, the framework moves underwriting from deterministic point-estimation toward probabilistic decision intelligence suitable for institutional investment committees.
