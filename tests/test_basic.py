"""
Basic unit tests for the portfolio-optimization source modules.
Run with: pytest tests/ -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import data_loader, eda, backtest


@pytest.fixture
def sample_prices():
    dates = pd.bdate_range("2023-01-01", periods=300)
    rng = np.random.default_rng(42)
    tsla = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, len(dates))))
    bnd = 80 * np.exp(np.cumsum(rng.normal(0.0001, 0.003, len(dates))))
    spy = 300 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(dates))))
    return pd.DataFrame({"TSLA": tsla, "BND": bnd, "SPY": spy}, index=dates)


@pytest.fixture
def sample_returns(sample_prices):
    return data_loader.compute_daily_returns(sample_prices)


def test_compute_daily_returns_shape(sample_prices, sample_returns):
    assert len(sample_returns) == len(sample_prices) - 1
    assert list(sample_returns.columns) == list(sample_prices.columns)


def test_combine_asset_data():
    dates = pd.bdate_range("2023-01-01", periods=10)
    data = {
        "TSLA": pd.DataFrame({"Adj Close": np.arange(10, dtype=float)}, index=dates),
        "SPY": pd.DataFrame({"Adj Close": np.arange(10, 20, dtype=float)}, index=dates),
    }
    combined = data_loader.combine_asset_data(data)
    assert list(combined.columns) == ["TSLA", "SPY"]
    assert combined.shape == (10, 2)


def test_adf_test_returns_expected_keys(sample_returns):
    result = eda.adf_test(sample_returns["TSLA"], label="TSLA_return")
    for key in ["adf_statistic", "p_value", "verdict", "label"]:
        assert key in result
    assert result["verdict"] in ("stationary", "non-stationary")


def test_returns_are_more_stationary_than_prices(sample_prices, sample_returns):
    price_result = eda.adf_test(sample_prices["TSLA"], label="TSLA_price")
    return_result = eda.adf_test(sample_returns["TSLA"], label="TSLA_return")
    assert return_result["p_value"] <= price_result["p_value"] + 1e-6


def test_value_at_risk_positive(sample_returns):
    var = eda.value_at_risk(sample_returns["TSLA"], confidence=0.95)
    assert var > 0


def test_sharpe_ratio_is_finite(sample_returns):
    sr = eda.sharpe_ratio(sample_returns["SPY"])
    assert np.isfinite(sr)


def test_detect_outliers_subset(sample_returns):
    outliers = eda.detect_outliers(sample_returns["TSLA"], n_std=3.0)
    assert isinstance(outliers, pd.Series)
    assert set(outliers.index).issubset(set(sample_returns.index))


def test_simulate_fixed_weights_no_rebalance(sample_returns):
    weights = {"TSLA": 0.2, "BND": 0.3, "SPY": 0.5}
    port_returns = backtest.simulate_fixed_weights(sample_returns, weights, rebalance=False)
    assert len(port_returns) == len(sample_returns)


def test_simulate_fixed_weights_rebalance(sample_returns):
    weights = {"TSLA": 0.2, "BND": 0.3, "SPY": 0.5}
    port_returns = backtest.simulate_fixed_weights(sample_returns, weights, rebalance=True)
    assert len(port_returns) == len(sample_returns)


def test_performance_metrics_keys(sample_returns):
    metrics = backtest.performance_metrics(sample_returns["SPY"])
    for key in ["total_return_pct", "annualized_return_pct", "sharpe_ratio", "max_drawdown_pct"]:
        assert key in metrics


def test_performance_metrics_drawdown_non_positive(sample_returns):
    metrics = backtest.performance_metrics(sample_returns["TSLA"])
    assert metrics["max_drawdown_pct"] <= 0
