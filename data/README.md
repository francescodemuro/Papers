# Data Documentation

## Included files
- `raw/fred_macro_sample.csv`: FRED-style macro indicators (policy rate, mortgage rate, CPI, unemployment, GDP growth).
- `raw/zillow_zhvi_sample.csv`: ZHVI/rent-index style metro panel (Austin, Phoenix, Tampa).

## Source notes
These files are compact, project-bundled samples shaped from publicly reported series patterns for reproducibility in constrained environments.

## How hybrid data is created
`build_hybrid_transaction_dataset` combines:
1. real-index anchors (`zhvi`, `rent_index`, macro columns), and
2. structural assumptions for property-level heterogeneity (size, quality, age, location).

This design preserves realism while enabling full end-to-end modeling without external API calls.

## Limitations
- Included samples are reduced in scope and frequency.
- For production deployment, connect directly to full FRED/Zillow pipelines and transaction-level county assessor/MLS data.
