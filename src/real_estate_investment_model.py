"""Investment-grade real estate decision engine.

This module upgrades the project from an academic demo to a practical
capital-allocation workflow with:
- hybrid real+synthetic data construction,
- walk-forward model validation,
- correlated regime-based Monte Carlo,
- realistic levered cash-flow analytics,
- multi-case investment recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------
# Data layer
# -----------------------------


def load_real_macro_data(path: str = "data/raw/fred_macro_sample.csv") -> pd.DataFrame:
    """Load macro data (proxy real series from FRED-style file).

    Expected columns: date, fed_funds, mortgage_30y, cpi_yoy, unemployment, gdp_yoy.
    """
    macro = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "fed_funds", "mortgage_30y", "cpi_yoy", "unemployment", "gdp_yoy"}
    missing = required - set(macro.columns)
    if missing:
        raise ValueError(f"Missing macro columns: {missing}")
    macro = macro.sort_values("date").reset_index(drop=True)
    return macro


def load_real_housing_index(path: str = "data/raw/zillow_zhvi_sample.csv") -> pd.DataFrame:
    """Load metro-level housing index data (ZHVI-style sample).

    Expected columns: date, metro, zhvi, rent_index.
    """
    zhvi = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "metro", "zhvi", "rent_index"}
    missing = required - set(zhvi.columns)
    if missing:
        raise ValueError(f"Missing housing columns: {missing}")
    return zhvi.sort_values(["metro", "date"]).reset_index(drop=True)


def build_hybrid_transaction_dataset(
    n_properties_per_metro: int = 180,
    seed: int = 42,
    macro_path: str = "data/raw/fred_macro_sample.csv",
    housing_path: str = "data/raw/zillow_zhvi_sample.csv",
) -> pd.DataFrame:
    """Construct transaction-like panel using real macro + real housing indices + structural assumptions."""
    rng = np.random.default_rng(seed)
    macro = load_real_macro_data(macro_path)
    housing = load_real_housing_index(housing_path)

    rows: List[pd.DataFrame] = []
    metros = housing["metro"].unique()

    for metro in metros:
        mdf = housing[housing["metro"] == metro].copy()
        mdf = mdf.merge(macro, on="date", how="left")

        n_t = len(mdf)
        n_obs = n_t * n_properties_per_metro

        size = rng.normal(1650, 500, n_obs).clip(500, 5000)
        bedrooms = rng.integers(1, 6, n_obs)
        bathrooms = (bedrooms - 0.1 + rng.normal(0, 0.5, n_obs)).clip(1, 5)
        age = rng.integers(0, 80, n_obs)
        school = rng.normal(72, 12, n_obs).clip(30, 100)
        transit = rng.normal(68, 15, n_obs).clip(15, 100)
        crime = rng.normal(50, 14, n_obs).clip(10, 95)
        dist_cbd = rng.gamma(2.2, 4.2, n_obs).clip(0.5, 45)

        repeated = mdf.loc[mdf.index.repeat(n_properties_per_metro)].reset_index(drop=True)
        loc_score = 0.40 * school + 0.25 * transit - 0.35 * crime - 0.08 * dist_cbd

        sale_price = (
            repeated["zhvi"].values
            * np.exp(
                0.00034 * (size - 1500)
                + 0.045 * (bedrooms - 3)
                + 0.03 * (bathrooms - 2)
                - 0.006 * age
                + 0.0035 * loc_score
                + rng.normal(0, 0.10, n_obs)
            )
        ).clip(80_000, None)

        annual_rent = (
            repeated["rent_index"].values * 12
            * (1 + 0.00011 * (size - 1500) + 0.03 * (bedrooms - 3) - 0.004 * age)
            * np.exp(rng.normal(0, 0.06, n_obs))
        ).clip(7_000, None)

        frame = pd.DataFrame(
            {
                "date": repeated["date"].values,
                "metro": metro,
                "size_sqft": size,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "age_years": age,
                "school_score": school,
                "transit_score": transit,
                "crime_rate_index": crime,
                "distance_to_cbd_km": dist_cbd,
                "fed_funds": repeated["fed_funds"].values,
                "mortgage_rate": repeated["mortgage_30y"].values,
                "inflation_rate": repeated["cpi_yoy"].values,
                "unemployment_rate": repeated["unemployment"].values,
                "gdp_growth": repeated["gdp_yoy"].values,
                "sale_price": sale_price,
                "annual_rent": annual_rent,
            }
        )
        frame["cap_rate"] = frame["annual_rent"] / frame["sale_price"]
        rows.append(frame)

    out = pd.concat(rows, ignore_index=True)
    out["year"] = out["date"].dt.year
    return out


def create_model_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    X = df.copy()
    X["log_size"] = np.log(X["size_sqft"])
    X["age_sq"] = X["age_years"] ** 2
    X["loc_composite"] = 0.45 * X["school_score"] + 0.30 * X["transit_score"] - 0.35 * X["crime_rate_index"]
    X["size_x_loc"] = X["size_sqft"] * X["loc_composite"]
    X["rent_to_price"] = X["annual_rent"] / X["sale_price"]

    metro_dummies = pd.get_dummies(X["metro"], prefix="metro", drop_first=True)
    X = pd.concat([X, metro_dummies], axis=1)

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
        "fed_funds",
        "mortgage_rate",
        "inflation_rate",
        "unemployment_rate",
        "gdp_growth",
        "cap_rate",
        "loc_composite",
        "size_x_loc",
        "rent_to_price",
    ] + list(metro_dummies.columns)

    y = np.log(X["sale_price"])
    return X[features], y, features


# -----------------------------
# Walk-forward ML evaluation
# -----------------------------


def walk_forward_validation(
    df: pd.DataFrame,
    min_train_years: int = 3,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Time-based walk-forward validation to test generalization across cycles."""
    x_all, y_all, features = create_model_matrix(df)
    temp = df[["date", "year"]].copy()
    temp = pd.concat([temp.reset_index(drop=True), x_all.reset_index(drop=True), y_all.rename("target")], axis=1)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.85,
            random_state=seed,
        ),
    }

    years = sorted(temp["year"].unique())
    folds = []
    best_by_model: Dict[str, List[float]] = {k: [] for k in models}

    for idx in range(min_train_years, len(years)):
        train_years = years[:idx]
        test_year = years[idx]

        train = temp[temp["year"].isin(train_years)]
        test = temp[temp["year"] == test_year]

        X_train = train[features]
        y_train = train["target"]
        X_test = test[features]
        y_test = test["target"]

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            r2 = float(r2_score(y_test, pred))
            folds.append({"model": name, "test_year": test_year, "rmse": rmse, "r2": r2})
            best_by_model[name].append(rmse)

    fold_df = pd.DataFrame(folds)
    score = fold_df.groupby("model")["rmse"].mean().sort_values()
    best_model_name = score.index[0]
    final_model = models[best_model_name]
    final_model.fit(x_all, y_all)

    summary = (
        fold_df.groupby("model")
        .agg(mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"), mean_r2=("r2", "mean"))
        .sort_values("mean_rmse")
        .reset_index()
    )
    return summary, {"best_model_name": best_model_name, "best_model": final_model, "features": features, "folds": fold_df}


# -----------------------------
# Financial engine
# -----------------------------


def npv(rate: float, cashflows: np.ndarray) -> float:
    t = np.arange(len(cashflows))
    return float(np.sum(cashflows / (1 + rate) ** t))


def irr(cashflows: np.ndarray, low: float = -0.90, high: float = 1.50) -> float:
    def f(r: float) -> float:
        return npv(r, cashflows)

    fl, fh = f(low), f(high)
    if fl * fh > 0:
        return np.nan

    for _ in range(120):
        mid = 0.5 * (low + high)
        fm = f(mid)
        if abs(fm) < 1e-9:
            return mid
        if fl * fm < 0:
            high, fh = mid, fm
        else:
            low, fl = mid, fm
    return 0.5 * (low + high)


@dataclass
class InvestmentCase:
    name: str
    purchase_price: float
    ltv: float
    initial_rate: float
    interest_only_years: int
    amort_years: int
    hold_years: int
    closing_cost_ratio: float
    renovation_cost: float
    annual_gross_rent: float
    rent_growth: float
    vacancy_rate: float
    opex_ratio: float
    exit_cap_rate: float
    sale_cost_ratio: float
    discount_rate: float
    hurdle_irr: float = 0.12


def annual_payment(principal: float, rate: float, years: int) -> float:
    if years <= 0:
        return principal
    if rate <= 1e-12:
        return principal / years
    g = (1 + rate) ** years
    return principal * (rate * g) / (g - 1)


def build_cashflow_case(
    case: InvestmentCase,
    price_growth: np.ndarray,
    rent_growth_shock: np.ndarray,
    interest_rates: np.ndarray,
) -> Dict[str, np.ndarray]:
    T = case.hold_years
    assert len(price_growth) == T and len(rent_growth_shock) == T and len(interest_rates) == T

    debt = case.purchase_price * case.ltv
    equity = case.purchase_price * (1 - case.ltv)
    upfront = equity + case.renovation_cost + case.closing_cost_ratio * case.purchase_price

    outstanding = debt
    prop_value = case.purchase_price
    rent = case.annual_gross_rent

    cfs = np.zeros(T + 1)
    cfs[0] = -upfront

    for y in range(1, T + 1):
        prop_value *= 1 + price_growth[y - 1]
        rent *= 1 + case.rent_growth + rent_growth_shock[y - 1]

        egi = rent * (1 - case.vacancy_rate)
        opex = egi * case.opex_ratio
        noi = egi - opex

        rate = max(0.001, interest_rates[y - 1])
        if y <= case.interest_only_years:
            debt_service = outstanding * rate
            principal = 0.0
        else:
            remain = max(case.amort_years - (y - case.interest_only_years - 1), 1)
            debt_service = annual_payment(outstanding, rate, remain)
            principal = max(debt_service - outstanding * rate, 0.0)
        outstanding = max(outstanding - principal, 0.0)

        cfs[y] = noi - debt_service

    terminal_noi = rent * (1 - case.vacancy_rate) * (1 - case.opex_ratio)
    exit_value = terminal_noi / max(case.exit_cap_rate, 0.01)
    net_sale = exit_value * (1 - case.sale_cost_ratio) - outstanding
    cfs[-1] += net_sale

    cumulative = np.cumsum(cfs)
    pos = np.where(cumulative > 0)[0]
    payback = int(pos[0]) if len(pos) else np.nan

    return {
        "cashflows": cfs,
        "exit_value": exit_value,
        "net_sale": net_sale,
        "payback": payback,
    }


# -----------------------------
# Correlated regime Monte Carlo
# -----------------------------


@dataclass
class RegimeSpec:
    name: str
    prob: float
    mean_price_growth: float
    mean_rent_growth_shock: float
    mean_rate_shift: float


@dataclass
class DecisionPolicy:
    """Formal investment committee decision policy."""

    max_prob_negative_npv_invest: float = 0.35
    max_prob_negative_npv_conditional: float = 0.50
    max_prob_irr_below_hurdle_invest: float = 0.45
    min_median_npv_invest: float = 0.0
    min_median_npv_conditional: float = -25_000.0


def simulate_correlated_paths(
    n_sims: int,
    years: int,
    base_rate: float,
    corr_matrix: np.ndarray,
    vol_price: float,
    vol_rent: float,
    vol_rate: float,
    regimes: List[RegimeSpec],
    seed: int = 7,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr_matrix)

    regime_probs = np.array([r.prob for r in regimes], dtype=float)
    regime_probs = regime_probs / regime_probs.sum()

    price = np.zeros((n_sims, years))
    rent = np.zeros((n_sims, years))
    rates = np.zeros((n_sims, years))
    regime_names = []

    for i in range(n_sims):
        reg = regimes[rng.choice(len(regimes), p=regime_probs)]
        regime_names.append(reg.name)

        z = rng.normal(size=(years, 3)) @ L.T
        price[i] = reg.mean_price_growth + vol_price * z[:, 0]
        rent[i] = reg.mean_rent_growth_shock + vol_rent * z[:, 1]

        rt = np.zeros(years)
        rt[0] = base_rate + reg.mean_rate_shift + vol_rate * z[0, 2]
        for t in range(1, years):
            rt[t] = 0.55 * rt[t - 1] + 0.45 * (base_rate + reg.mean_rate_shift) + vol_rate * z[t, 2]
        rates[i] = np.clip(rt, 0.005, 0.16)

    return {"price": price, "rent": rent, "rates": rates, "regime": np.array(regime_names)}


def monte_carlo_case(case: InvestmentCase, n_sims: int = 15000, seed: int = 42) -> pd.DataFrame:
    regimes = [
        RegimeSpec("Bull", 0.25, 0.055, 0.020, -0.010),
        RegimeSpec("Base", 0.55, 0.030, 0.005, 0.000),
        RegimeSpec("Bear", 0.20, -0.010, -0.010, 0.015),
    ]

    corr = np.array(
        [
            [1.00, 0.55, -0.45],
            [0.55, 1.00, -0.35],
            [-0.45, -0.35, 1.00],
        ]
    )

    paths = simulate_correlated_paths(
        n_sims=n_sims,
        years=case.hold_years,
        base_rate=case.initial_rate,
        corr_matrix=corr,
        vol_price=0.08,
        vol_rent=0.035,
        vol_rate=0.012,
        regimes=regimes,
        seed=seed,
    )

    rows = []
    for i in range(n_sims):
        cf = build_cashflow_case(case, paths["price"][i], paths["rent"][i], paths["rates"][i])["cashflows"]
        r_irr = irr(cf)
        r_npv = npv(case.discount_rate, cf)
        rows.append(
            {
                "case": case.name,
                "regime": paths["regime"][i],
                "irr": r_irr,
                "npv": r_npv,
                "avg_rate": float(paths["rates"][i].mean()),
            }
        )

    sim = pd.DataFrame(rows)
    var_5 = sim["npv"].quantile(0.05)
    cvar_5 = sim.loc[sim["npv"] <= var_5, "npv"].mean()

    sim.attrs["risk_summary"] = {
        "prob_npv_negative": float((sim["npv"] < 0).mean()),
        "prob_irr_below_hurdle": float((sim["irr"] < case.hurdle_irr).mean()),
        "var_5_npv": float(var_5),
        "cvar_5_npv": float(cvar_5),
    }
    return sim


def evaluate_investment_decision(
    sim: pd.DataFrame,
    hurdle_irr: float,
    policy: DecisionPolicy | None = None,
) -> Dict[str, float | str]:
    """Return formal decision label using risk-adjusted committee policy."""
    if policy is None:
        policy = DecisionPolicy()

    rs = sim.attrs.get("risk_summary", {})
    median_irr = float(sim["irr"].median())
    median_npv = float(sim["npv"].median())
    p_npv_neg = float(rs.get("prob_npv_negative", 1.0))
    p_irr_fail = float(rs.get("prob_irr_below_hurdle", 1.0))

    if (
        median_npv >= policy.min_median_npv_invest
        and p_npv_neg <= policy.max_prob_negative_npv_invest
        and p_irr_fail <= policy.max_prob_irr_below_hurdle_invest
        and median_irr >= hurdle_irr
    ):
        rec = "Invest"
    elif (
        median_npv >= policy.min_median_npv_conditional
        and p_npv_neg <= policy.max_prob_negative_npv_conditional
    ):
        rec = "Conditional Invest"
    else:
        rec = "Do Not Invest"

    return {
        "median_irr": median_irr,
        "median_npv": median_npv,
        "recommendation": rec,
        "decision_policy": (
            f"Invest if median NPV≥{policy.min_median_npv_invest:,.0f}, "
            f"P(NPV<0)≤{policy.max_prob_negative_npv_invest:.0%}, "
            f"P(IRR<hurdle)≤{policy.max_prob_irr_below_hurdle_invest:.0%}, "
            f"and median IRR≥hurdle."
        ),
        **rs,
    }


def default_investment_cases() -> List[InvestmentCase]:
    return [
        InvestmentCase(
            name="Case A - Core Stabilized",
            purchase_price=1_250_000,
            ltv=0.60,
            initial_rate=0.055,
            interest_only_years=1,
            amort_years=25,
            hold_years=7,
            closing_cost_ratio=0.028,
            renovation_cost=80_000,
            annual_gross_rent=120_000,
            rent_growth=0.028,
            vacancy_rate=0.05,
            opex_ratio=0.34,
            exit_cap_rate=0.058,
            sale_cost_ratio=0.045,
            discount_rate=0.095,
            hurdle_irr=0.11,
        ),
        InvestmentCase(
            name="Case B - Value Add",
            purchase_price=980_000,
            ltv=0.70,
            initial_rate=0.060,
            interest_only_years=2,
            amort_years=25,
            hold_years=7,
            closing_cost_ratio=0.030,
            renovation_cost=170_000,
            annual_gross_rent=96_000,
            rent_growth=0.035,
            vacancy_rate=0.07,
            opex_ratio=0.37,
            exit_cap_rate=0.062,
            sale_cost_ratio=0.050,
            discount_rate=0.105,
            hurdle_irr=0.13,
        ),
        InvestmentCase(
            name="Case C - Opportunistic",
            purchase_price=1_500_000,
            ltv=0.75,
            initial_rate=0.064,
            interest_only_years=2,
            amort_years=30,
            hold_years=7,
            closing_cost_ratio=0.032,
            renovation_cost=250_000,
            annual_gross_rent=132_000,
            rent_growth=0.038,
            vacancy_rate=0.09,
            opex_ratio=0.39,
            exit_cap_rate=0.067,
            sale_cost_ratio=0.052,
            discount_rate=0.115,
            hurdle_irr=0.15,
        ),
    ]
