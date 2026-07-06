"""
Modern Portfolio Theory (MPT) utilities: expected returns, covariance
matrix, Efficient Frontier generation, and identification of the
Maximum Sharpe Ratio and Minimum Volatility portfolios.
"""

import logging

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models
from pypfopt import expected_returns as pfopt_expected_returns

logger = logging.getLogger(__name__)


def build_expected_returns(returns: pd.DataFrame, tsla_forecast_annual_return: float) -> pd.Series:
    """
    Build the expected-returns vector used for optimization.

    TSLA uses the forecasted annualized return from the best Task-2 model.
    BND and SPY use their historical annualized mean daily return as a proxy.
    """
    if returns is None or returns.empty:
        raise ValueError("build_expected_returns requires a non-empty returns DataFrame")

    hist_annual = pfopt_expected_returns.mean_historical_return(
        returns, returns_data=True, compounding=True, frequency=252
    )

    mu = hist_annual.copy()
    mu["TSLA"] = tsla_forecast_annual_return
    return mu


def build_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized sample covariance matrix of daily returns."""
    if returns is None or returns.empty:
        raise ValueError("build_covariance_matrix requires a non-empty returns DataFrame")

    return risk_models.sample_cov(returns, returns_data=True, frequency=252)


def efficient_frontier_points(mu: pd.Series, cov: pd.DataFrame, n_points: int = 60):
    """
    Sweep target returns across the feasible range and solve for the
    minimum-volatility portfolio at each target, tracing the Efficient
    Frontier. Returns arrays of (volatilities, returns, weights_list).
    """
    if mu is None or mu.empty:
        raise ValueError("efficient_frontier_points requires a non-empty expected returns series")
    if cov is None or cov.empty:
        raise ValueError("efficient_frontier_points requires a non-empty covariance matrix")
    if n_points <= 0:
        raise ValueError("efficient_frontier_points requires n_points to be positive")

    ef = EfficientFrontier(mu, cov)
    min_ret = mu.min()
    max_ret = mu.max()
    if np.isclose(min_ret, max_ret):
        raise RuntimeError("Efficient frontier requires a non-trivial expected return range")

    targets = np.linspace(min_ret + 1e-4, max_ret - 1e-4, n_points)

    vols, rets, weights_list = [], [], []
    for t in targets:
        try:
            ef_i = EfficientFrontier(mu, cov)
            ef_i.efficient_return(target_return=t)
            perf = ef_i.portfolio_performance()
            vols.append(perf[1])
            rets.append(perf[0])
            weights_list.append(ef_i.clean_weights())
        except Exception as exc:
            logger.warning("Skipping efficient frontier target %.6f because optimization failed: %s", t, exc)
            continue
    if not vols:
        raise RuntimeError("Efficient frontier optimization failed for all target returns")
    return np.array(vols), np.array(rets), weights_list


def max_sharpe_portfolio(mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float = 0.02):
    """Solve for the tangency (max Sharpe ratio) portfolio."""
    try:
        ef = EfficientFrontier(mu, cov)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        return weights, perf
    except Exception as exc:
        raise RuntimeError("Max Sharpe portfolio optimization failed") from exc


def min_volatility_portfolio(mu: pd.Series, cov: pd.DataFrame):
    """Solve for the global minimum-volatility portfolio."""
    try:
        ef = EfficientFrontier(mu, cov)
        ef.min_volatility()
        weights = ef.clean_weights()
        perf = ef.portfolio_performance()
        return weights, perf
    except Exception as exc:
        raise RuntimeError("Minimum-volatility portfolio optimization failed") from exc
