"""
Classical statistical forecasting: ARIMA/SARIMA model selection, fitting,
and forecasting for Tesla (TSLA) closing prices.
"""

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA


def find_best_order(train: pd.Series, seasonal: bool = False, m: int = 5):
    """
    Use pmdarima's auto_arima to search for the best (p, d, q) [and seasonal
    (P, D, Q, m) if seasonal=True] parameters by AIC minimization.
    """
    model = pm.auto_arima(
        train,
        start_p=0, start_q=0, max_p=5, max_q=5, max_d=2,
        seasonal=seasonal, m=m if seasonal else 1,
        stepwise=True, suppress_warnings=True, error_action="ignore",
        trace=False, with_intercept=True,
    )
    order = model.order
    seasonal_order = model.seasonal_order if seasonal else None
    return order, seasonal_order, model


def fit_arima(train: pd.Series, order):
    """Fit a statsmodels ARIMA model with a given (p, d, q) order, allowing drift/intercept."""
    trend = "t" if order[1] >= 1 else "c"
    model = ARIMA(train, order=order, trend=trend)
    return model.fit()


def forecast_arima(fitted_model, steps: int, alpha: float = 0.05):
    """
    Generate point forecasts and confidence intervals for `steps` periods.
    Returns (forecast_series, lower_ci, upper_ci).
    """
    result = fitted_model.get_forecast(steps=steps)
    mean = result.predicted_mean
    ci = result.conf_int(alpha=alpha)
    lower = ci.iloc[:, 0]
    upper = ci.iloc[:, 1]
    return mean, lower, upper
