# Data Documentation and Credibility Notes

## Included datasets
- `raw/fred_macro_sample.csv`
  - Variables: fed funds proxy, 30Y mortgage rate proxy, CPI YoY, unemployment, GDP YoY.
  - Intended to represent the macro state vector that drives financing and demand conditions.
- `raw/zillow_zhvi_sample.csv`
  - Variables: metro-level home value index proxy (`zhvi`) and rent index proxy.
  - Markets included: Austin, Phoenix, Tampa.

## Source rationale
The project uses compact, version-controlled samples that mirror publicly reported FRED and ZHVI series behavior so the full workflow is reproducible in a constrained environment.

## Why this supports credible decisions
- Real macro and housing-directional information anchors valuation and stress testing.
- Synthetic property-level heterogeneity is only used to complete cross-sectional underwriting features, not to invent macro regime behavior.

## Limitations and investment implications
1. **Coverage limitation:** three metros and semiannual observations understate regional dispersion.
   - Decision impact: treat outputs as screening and policy calibration, not market-by-market final IC approval.
2. **Calibration limitation:** regime probabilities and correlations are practitioner-calibrated.
   - Decision impact: stress-test recommendations under alternate regime priors before committing capital.
3. **Data granularity limitation:** no parcel-level capex history, taxes, or lease rolls.
   - Decision impact: incorporate deal-level due diligence overlays prior to execution.

## Production extension path
For deployment, connect directly to full FRED/Zillow feeds and property-level transaction/lease data, then retrain walk-forward models quarterly with formal model risk governance.
