"""
Classical statistical forecasting: ARIMA/SARIMA model selection, fitting,
and forecasting for Tesla (TSLA) closing prices.
"""

import logging

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


def find_best_order(train: pd.Series, seasonal: bool = False, m: int = 5):
    """
    Use pmdarima's auto_arima to search for the best (p, d, q) [and seasonal
    (P, D, Q, m) if seasonal=True] parameters by AIC minimization.
    """
    if train is None or train.empty:
        raise ValueError("find_best_order requires a non-empty training series")

    logger.info("Searching for best ARIMA order on %d observations", len(train))
    try:
        model = pm.auto_arima(
            train,
            start_p=0, start_q=0, max_p=5, max_q=5, max_d=2,
            seasonal=seasonal, m=m if seasonal else 1,
            stepwise=True, suppress_warnings=True, error_action="ignore",
            trace=False, with_intercept=True,
        )
    except Exception as exc:
        raise RuntimeError("ARIMA order search failed during auto_arima selection") from exc

    order = model.order
    seasonal_order = model.seasonal_order if seasonal else None
    logger.info("Selected ARIMA order %s", order)
    return order, seasonal_order, model


def fit_arima(train: pd.Series, order):
    """Fit a statsmodels ARIMA model with a given (p, d, q) order, allowing drift/intercept."""
    if train is None or train.empty:
        raise ValueError("fit_arima requires a non-empty training series")
    if order is None or len(order) != 3:
        raise ValueError(f"fit_arima expected a 3-tuple ARIMA order, got {order!r}")

    trend = "t" if order[1] >= 1 else "c"
    logger.info("Fitting ARIMA%s model with trend '%s'", order, trend)
    try:
        model = ARIMA(train, order=order, trend=trend)
        return model.fit()
    except Exception as exc:
        raise RuntimeError(f"ARIMA fitting failed for order {order}") from exc


def forecast_arima(fitted_model, steps: int, alpha: float = 0.05):
    """
    Generate point forecasts and confidence intervals for `steps` periods.
    Returns (forecast_series, lower_ci, upper_ci).
    """
    if steps <= 0:
        raise ValueError("forecast_arima requires steps to be a positive integer")

    try:
        result = fitted_model.get_forecast(steps=steps)
        mean = result.predicted_mean
        ci = result.conf_int(alpha=alpha)
        lower = ci.iloc[:, 0]
        upper = ci.iloc[:, 1]
        logger.info("Generated ARIMA forecast for %d steps", steps)
        return mean, lower, upper
    except Exception as exc:
        raise RuntimeError(f"ARIMA forecasting failed for horizon {steps}") from exc
