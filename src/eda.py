"""
Exploratory data analysis utilities: trend/volatility visualization,
stationarity testing, outlier detection, and foundational risk metrics
(VaR, Sharpe Ratio).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def rolling_stats(series: pd.Series, window: int = 30):
    """Return rolling mean and rolling standard deviation for a series."""
    return series.rolling(window).mean(), series.rolling(window).std()


def detect_outliers(returns: pd.Series, n_std: float = 3.0) -> pd.Series:
    """Flag daily returns more than n_std standard deviations from the mean."""
    mu, sigma = returns.mean(), returns.std()
    z = (returns - mu) / sigma
    return returns[z.abs() > n_std]


def adf_test(series: pd.Series, label: str = "") -> dict:
    """
    Run the Augmented Dickey-Fuller test for stationarity.

    Returns a dict with test statistic, p-value, critical values, and a
    plain-language verdict. A p-value < 0.05 rejects the null hypothesis
    of a unit root -> series is considered stationary.
    """
    series = series.dropna()
    result = adfuller(series, autolag="AIC")
    verdict = "stationary" if result[1] < 0.05 else "non-stationary"
    return {
        "label": label,
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "verdict": verdict,
    }


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical (empirical) Value at Risk at the given confidence level.
    Returned as a positive number representing the loss threshold.
    """
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe Ratio from daily returns.

    risk_free_rate is annualized; converted to a daily rate for the
    excess-return calculation, then re-annualized.
    """
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - daily_rf
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def summary_statistics(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Build a per-asset summary table: annualized return, volatility, Sharpe, VaR, skew, kurtosis."""
    rows = []
    for col in prices.columns:
        r = returns[col].dropna()
        ann_return = (1 + r.mean()) ** 252 - 1
        ann_vol = r.std() * np.sqrt(252)
        rows.append({
            "asset": col,
            "start_price": prices[col].iloc[0],
            "end_price": prices[col].iloc[-1],
            "total_return_pct": (prices[col].iloc[-1] / prices[col].iloc[0] - 1) * 100,
            "annualized_return_pct": ann_return * 100,
            "annualized_volatility_pct": ann_vol * 100,
            "sharpe_ratio": sharpe_ratio(r),
            "var_95_daily_pct": value_at_risk(r, 0.95) * 100,
            "skew": r.skew(),
            "kurtosis": r.kurtosis(),
        })
    return pd.DataFrame(rows).set_index("asset")
