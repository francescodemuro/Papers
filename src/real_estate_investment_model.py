"""Core quantitative framework for data-driven real estate investment analysis.

This module provides:
1) Synthetic but economically grounded real-estate panel data generation
2) Feature engineering utilities
3) Machine-learning model training/evaluation with validation split
4) Levered cash-flow modeling and valuation metrics
5) Monte Carlo risk simulation with stochastic growth + interest rates
6) Sensitivity analysis wrappers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return (arr - arr.mean()) / (arr.std() + 1e-8)


def generate_synthetic_real_estate_data(
    n_properties: int = 2500,
    years: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic panel-like dataset of property transactions."""
    rng = np.random.default_rng(seed)

    n_obs = n_properties * years
    property_id = np.repeat(np.arange(n_properties), years)
    year_idx = np.tile(np.arange(years), n_properties)

    base_size = rng.normal(1650, 550, n_properties).clip(450, 5500)
    beds = rng.integers(1, 6, n_properties)
    baths = (beds - 0.2 + rng.normal(0, 0.5, n_properties)).clip(1, 5)
    age0 = rng.integers(0, 70, n_properties)
    dist_cbd = rng.gamma(shape=2.3, scale=4.5, size=n_properties).clip(0.4, 40)
    school_idx = rng.normal(75, 12, n_properties).clip(30, 100)
    transit_idx = rng.normal(70, 15, n_properties).clip(20, 100)
    crime_idx = rng.normal(50, 15, n_properties).clip(10, 95)

    size = np.repeat(base_size, years) * (1 + rng.normal(0, 0.01, n_obs))
    bedrooms = np.repeat(beds, years)
    bathrooms = np.repeat(baths, years)
    age = np.repeat(age0, years) + year_idx
    distance_to_cbd = np.repeat(dist_cbd, years)
    school_score = np.repeat(school_idx, years)
    transit_score = np.repeat(transit_idx, years)
    crime_rate_index = np.repeat(crime_idx, years)

    t = np.arange(years)
    mortgage_rate_year = 0.028 + 0.012 * np.sin(t / 1.6) + 0.002 * t + rng.normal(0, 0.002, years)
    unemployment_year = 0.045 + 0.01 * np.cos(t / 1.4) + rng.normal(0, 0.002, years)
    inflation_year = 0.021 + 0.005 * np.sin(t / 1.9 + 0.5) + rng.normal(0, 0.0015, years)
    gdp_growth_year = 0.02 + 0.006 * np.cos(t / 2.1) + rng.normal(0, 0.0018, years)

    mortgage_rate = np.take(mortgage_rate_year, year_idx)
    unemployment_rate = np.take(unemployment_year, year_idx)
    inflation_rate = np.take(inflation_year, year_idx)
    gdp_growth = np.take(gdp_growth_year, year_idx)

    loc_quality = (
        0.35 * _normalize(school_score)
        + 0.25 * _normalize(transit_score)
        - 0.30 * _normalize(crime_rate_index)
        - 0.20 * _normalize(distance_to_cbd)
    )

    cycle = 0.04 * np.sin(year_idx / 1.7) + 0.02 * np.cos(year_idx / 0.9)
    trend = 0.025 * year_idx

    log_price = (
        10.85
        + 0.00038 * size
        + 0.052 * bedrooms
        + 0.045 * bathrooms
        - 0.008 * age
        + 0.34 * loc_quality
        + trend
        + cycle
        - 3.8 * mortgage_rate
        - 1.9 * unemployment_rate
        + 1.4 * inflation_rate
        + 2.8 * gdp_growth
        + rng.normal(0, 0.12, n_obs)
    )

    sale_price = np.exp(log_price)
    monthly_rent = (
        0.0042 * sale_price
        + 0.18 * size
        + 55 * bedrooms
        - 20 * age
        + rng.normal(0, 260, n_obs)
    ).clip(500, None)
    cap_rate = (monthly_rent * 12 / sale_price).clip(0.02, 0.12)

    return pd.DataFrame(
        {
            "property_id": property_id,
            "year_index": year_idx,
            "size_sqft": size,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age,
            "distance_to_cbd_km": distance_to_cbd,
            "school_score": school_score,
            "transit_score": transit_score,
            "crime_rate_index": crime_rate_index,
            "mortgage_rate": mortgage_rate,
            "unemployment_rate": unemployment_rate,
            "inflation_rate": inflation_rate,
            "gdp_growth": gdp_growth,
            "monthly_rent": monthly_rent,
            "cap_rate": cap_rate,
            "sale_price": sale_price,
        }
    )


def create_model_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Feature engineering for pricing model."""
    X = df.copy()
    X["log_size"] = np.log(X["size_sqft"])
    X["size_x_school"] = X["size_sqft"] * X["school_score"] / 100.0
    X["age_sq"] = X["age_years"] ** 2
    X["bed_bath_ratio"] = X["bedrooms"] / (X["bathrooms"] + 1e-6)
    X["loc_composite"] = 0.4 * X["school_score"] + 0.3 * X["transit_score"] - 0.3 * X["crime_rate_index"]
    X["rent_to_price"] = X["monthly_rent"] * 12 / X["sale_price"]

    features = [
        "log_size",
        "bedrooms",
        "bathrooms",
        "age_years",
        "age_sq",
        "distance_to_cbd_km",
        "school_score",
        "transit_score",
        "crime_rate_index",
        "mortgage_rate",
        "unemployment_rate",
        "inflation_rate",
        "gdp_growth",
        "size_x_school",
        "bed_bath_ratio",
        "loc_composite",
        "cap_rate",
        "rent_to_price",
        "year_index",
    ]

    y = np.log(X["sale_price"])
    return X[features], y, features


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object], tuple]:
    """Train and compare multiple regression models with train/val/test split."""
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=random_state)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.85,
            random_state=random_state,
        ),
    }

    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "r2_train": r2_score(y_train, pred_train),
                "r2_val": r2_score(y_val, pred_val),
                "r2_test": r2_score(y_test, pred_test),
                "rmse_train": np.sqrt(mean_squared_error(y_train, pred_train)),
                "rmse_val": np.sqrt(mean_squared_error(y_val, pred_val)),
                "rmse_test": np.sqrt(mean_squared_error(y_test, pred_test)),
            }
        )
        fitted[name] = model

    result = pd.DataFrame(rows).sort_values("rmse_val").reset_index(drop=True)
    splits = (X_train, X_val, X_test, y_train, y_val, y_test)
    return result, fitted, splits


def npv(rate: float, cashflows: np.ndarray) -> float:
    periods = np.arange(len(cashflows))
    return float(np.sum(cashflows / (1 + rate) ** periods))


def irr(cashflows: np.ndarray, low: float = -0.95, high: float = 2.0) -> float:
    """Compute IRR using bisection."""

    def f(r: float) -> float:
        return npv(r, cashflows)

    fl, fh = f(low), f(high)
    if fl * fh > 0:
        return np.nan

    for _ in range(120):
        mid = (low + high) / 2
        fm = f(mid)
        if abs(fm) < 1e-8:
            return mid
        if fl * fm < 0:
            high, fh = mid, fm
        else:
            low, fl = mid, fm
    return (low + high) / 2


@dataclass
class InvestmentAssumptions:
    holding_period_years: int = 7
    purchase_price: float = 850_000
    down_payment_ratio: float = 0.30
    annual_interest_rate: float = 0.056
    interest_rate_mean_reversion: float = 0.35
    long_run_interest_rate: float = 0.055
    interest_rate_vol: float = 0.012
    loan_amort_years: int = 30
    closing_cost_ratio: float = 0.03
    renovation_cost: float = 40_000
    annual_rent: float = 52_000
    rent_growth_mu: float = 0.025
    rent_growth_sigma: float = 0.03
    vacancy_rate: float = 0.06
    operating_expense_ratio: float = 0.32
    annual_price_growth_mu: float = 0.03
    annual_price_growth_sigma: float = 0.07
    sale_cost_ratio: float = 0.05
    discount_rate: float = 0.09


def _annual_mortgage_payment(principal: float, annual_rate: float, years: int) -> float:
    if years <= 0:
        return principal
    if abs(annual_rate) < 1e-12:
        return principal / years
    growth = (1 + annual_rate) ** years
    return principal * (annual_rate * growth) / (growth - 1)


def simulate_interest_rate_path(assump: InvestmentAssumptions, rng: np.random.Generator) -> np.ndarray:
    """Simulate annual variable mortgage rates using a mean-reverting process."""
    T = assump.holding_period_years
    rates = np.zeros(T)
    rates[0] = max(0.005, assump.annual_interest_rate + rng.normal(0, assump.interest_rate_vol / 2))
    for t in range(1, T):
        drift = assump.interest_rate_mean_reversion * (assump.long_run_interest_rate - rates[t - 1])
        shock = assump.interest_rate_vol * rng.normal()
        rates[t] = np.clip(rates[t - 1] + drift + shock, 0.005, 0.20)
    return rates


def build_cashflows(
    assump: InvestmentAssumptions,
    price_growth_path: np.ndarray,
    rent_growth_path: np.ndarray,
    interest_rate_path: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Build annual levered cash flows under dynamic macro/financing paths."""
    T = assump.holding_period_years
    assert len(price_growth_path) == T
    assert len(rent_growth_path) == T

    if interest_rate_path is None:
        interest_rate_path = np.full(T, assump.annual_interest_rate)
    assert len(interest_rate_path) == T

    purchase = assump.purchase_price
    debt = purchase * (1 - assump.down_payment_ratio)
    equity = purchase * assump.down_payment_ratio
    upfront = equity + assump.closing_cost_ratio * purchase + assump.renovation_cost

    outstanding = debt
    annual_rent = assump.annual_rent
    cf = np.zeros(T + 1)
    cf[0] = -upfront

    property_value = purchase
    for y in range(1, T + 1):
        rate_y = interest_rate_path[y - 1]
        property_value *= (1 + price_growth_path[y - 1])
        annual_rent *= (1 + rent_growth_path[y - 1])

        effective_gross_income = annual_rent * (1 - assump.vacancy_rate)
        operating_expenses = assump.operating_expense_ratio * effective_gross_income
        noi = effective_gross_income - operating_expenses

        remaining_term = max(assump.loan_amort_years - (y - 1), 1)
        annual_payment = _annual_mortgage_payment(outstanding, rate_y, remaining_term)
        interest = outstanding * rate_y
        principal = max(annual_payment - interest, 0)
        outstanding = max(outstanding - principal, 0)
        cf[y] = noi - annual_payment

    sale_proceeds = property_value * (1 - assump.sale_cost_ratio) - outstanding
    cf[-1] += sale_proceeds

    cumulative = np.cumsum(cf)
    positive_idx = np.where(cumulative > 0)[0]
    payback_period = int(positive_idx[0]) if len(positive_idx) > 0 else np.nan

    return {
        "cashflows": cf,
        "property_value_terminal": property_value,
        "sale_proceeds": sale_proceeds,
        "outstanding_balance": outstanding,
        "payback_period": payback_period,
    }


def monte_carlo_investment(
    assump: InvestmentAssumptions,
    n_sims: int = 10000,
    seed: int = 123,
) -> pd.DataFrame:
    """Monte Carlo simulation for IRR and NPV under uncertainty.

    Uncertain variables:
    - annual property price growth
    - annual rental growth
    - annual borrowing rate (mean-reverting)
    """
    rng = np.random.default_rng(seed)
    T = assump.holding_period_years

    records = []
    for _ in range(n_sims):
        price_growth = rng.normal(assump.annual_price_growth_mu, assump.annual_price_growth_sigma, T)
        rent_growth = rng.normal(assump.rent_growth_mu, assump.rent_growth_sigma, T)
        interest_path = simulate_interest_rate_path(assump, rng)

        cfs = build_cashflows(assump, price_growth, rent_growth, interest_path)
        cf = cfs["cashflows"]

        records.append(
            {
                "irr": irr(cf),
                "npv": npv(assump.discount_rate, cf),
                "total_return": (cf.sum() / -cf[0]) - 1,
                "terminal_value": cfs["property_value_terminal"],
                "sale_proceeds": cfs["sale_proceeds"],
                "avg_interest_rate": float(np.mean(interest_path)),
            }
        )

    return pd.DataFrame(records)


def run_sensitivity_grid(base: InvestmentAssumptions) -> pd.DataFrame:
    """Grid sensitivity analysis across key assumptions."""
    interest_grid = [base.annual_interest_rate - 0.015, base.annual_interest_rate, base.annual_interest_rate + 0.015]
    price_grid = [base.purchase_price * 0.9, base.purchase_price, base.purchase_price * 1.1]
    rent_grid = [base.annual_rent * 0.9, base.annual_rent, base.annual_rent * 1.1]

    rows = []
    for ir in interest_grid:
        for pp in price_grid:
            for rent in rent_grid:
                a = InvestmentAssumptions(**{**base.__dict__, "annual_interest_rate": ir, "purchase_price": pp, "annual_rent": rent})
                price_path = np.full(a.holding_period_years, a.annual_price_growth_mu)
                rent_path = np.full(a.holding_period_years, a.rent_growth_mu)
                rate_path = np.full(a.holding_period_years, a.annual_interest_rate)
                cf = build_cashflows(a, price_path, rent_path, rate_path)["cashflows"]
                rows.append(
                    {
                        "interest_rate": ir,
                        "purchase_price": pp,
                        "annual_rent": rent,
                        "irr": irr(cf),
                        "npv": npv(a.discount_rate, cf),
                    }
                )
    return pd.DataFrame(rows)
