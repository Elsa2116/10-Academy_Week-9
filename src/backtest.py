"""
Strategy backtesting utilities: simulate a fixed-weight (or monthly
rebalanced) portfolio against a static benchmark over a historical
holdout window, and compute standard performance metrics.
"""

import numpy as np
import pandas as pd


def simulate_fixed_weights(returns: pd.DataFrame, weights: dict, rebalance: bool = False) -> pd.Series:
    """
    Simulate cumulative portfolio returns given a dict of {asset: weight}.

    If rebalance=True, weights are reset to target at the start of each
    calendar month; otherwise weights drift with asset performance
    ("buy and hold").
    """
    assets = list(weights.keys())
    w = np.array([weights[a] for a in assets])
    r = returns[assets].copy()

    if not rebalance:
        cum_growth = (1 + r).cumprod()
        weighted = cum_growth.multiply(w, axis=1)
        portfolio_value = weighted.sum(axis=1)
        portfolio_returns = portfolio_value.pct_change().fillna(portfolio_value.iloc[0] - 1)
        return portfolio_returns

    portfolio_returns = []
    current_weights = w.copy()
    current_month = r.index[0].month
    for date, day_returns in r.iterrows():
        if date.month != current_month:
            current_weights = w.copy()
            current_month = date.month
        port_ret = np.dot(current_weights, day_returns.values)
        portfolio_returns.append(port_ret)
        growth = current_weights * (1 + day_returns.values)
        current_weights = growth / growth.sum()
    return pd.Series(portfolio_returns, index=r.index)


def performance_metrics(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> dict:
    """Compute total return, annualized return, Sharpe ratio, and max drawdown."""
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    n_years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - daily_rf
    sharpe = (excess.mean() / excess.std()) * np.sqrt(periods_per_year) if excess.std() > 0 else np.nan

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "total_return_pct": total_return * 100,
        "annualized_return_pct": annualized_return * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
    }
