"""
End-to-end pipeline runner for the Time Series Forecasting for Portfolio
Management Optimization project.

Executes all 5 tasks in sequence and writes every artifact (cleaned data,
metrics, models, and figures) to data/processed and reports/figures so
that the accompanying notebooks and final report can consume them.

Usage:
    python scripts/run_pipeline.py
"""

import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from src import data_loader, eda, arima_model, lstm_model, portfolio, backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

FIG_DIR = "reports/figures"
PROC_DIR = "data/processed"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
BACKTEST_START = "2025-01-01"


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved figure %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# TASK 1 — Preprocess and Explore the Data
# ═══════════════════════════════════════════════════════════════════════════
def task1_eda():
    print("\n=== TASK 1: Preprocess and Explore the Data ===")
    try:
        raw = data_loader.fetch_asset_data()
    except Exception as exc:
        raise RuntimeError("Task 1 failed while downloading historical asset data") from exc

    try:
        cleaned = data_loader.clean_asset_data(raw)
    except Exception as exc:
        raise RuntimeError("Task 1 failed while cleaning historical asset data") from exc

    quality_report = data_loader.data_quality_report(raw, cleaned)
    quality_report.to_csv(f"{PROC_DIR}/data_quality_report.csv", index=False)
    print(quality_report.to_string(index=False))

    prices = data_loader.combine_asset_data(cleaned, field="Adj Close")
    prices.to_csv(f"{PROC_DIR}/adj_close_prices.csv")
    returns = data_loader.compute_daily_returns(prices)
    returns.to_csv(f"{PROC_DIR}/daily_returns.csv")

    stats = eda.summary_statistics(prices, returns)
    stats.to_csv(f"{PROC_DIR}/summary_statistics.csv")
    print("\nSummary statistics:\n", stats.round(3).to_string())

    # Visualization 1: Closing price over time (all 3 assets, normalized)
    fig, ax = plt.subplots(figsize=(11, 5))
    normalized = prices / prices.iloc[0] * 100
    normalized.plot(ax=ax, linewidth=1.6)
    ax.set_title("Normalized Adjusted Close Price (2015-01-01 = 100)")
    ax.set_ylabel("Indexed Price")
    ax.set_xlabel("Date")
    ax.legend(title="Asset")
    save_fig(fig, "01_normalized_prices.png")

    # Visualization 2: Daily % change (volatility) for all assets
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for i, col in enumerate(returns.columns):
        axes[i].plot(returns.index, returns[col] * 100, linewidth=0.5, color=sns.color_palette()[i])
        axes[i].set_ylabel(f"{col}\nDaily % Change")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Daily Percentage Change (Volatility) by Asset", y=1.0)
    save_fig(fig, "02_daily_pct_change.png")

    # Visualization 3: Rolling mean & std for TSLA
    roll_mean, roll_std = eda.rolling_stats(returns["TSLA"], window=30)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(returns.index, returns["TSLA"] * 100, alpha=0.3, label="Daily Return", linewidth=0.6)
    ax.plot(roll_mean.index, roll_mean * 100, label="30-Day Rolling Mean", linewidth=1.8)
    ax.plot(roll_std.index, roll_std * 100, label="30-Day Rolling Std Dev", linewidth=1.8)
    ax.set_title("TSLA Daily Returns — Rolling Mean and Volatility (30-day window)")
    ax.set_ylabel("% Return")
    ax.legend()
    save_fig(fig, "03_tsla_rolling_volatility.png")

    # Visualization 4: Outlier detection on TSLA returns
    outliers = eda.detect_outliers(returns["TSLA"], n_std=3.0)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(returns.index, returns["TSLA"] * 100, linewidth=0.6, color="steelblue", label="Daily Return")
    ax.scatter(outliers.index, outliers * 100, color="crimson", s=28, zorder=5, label=f"Outliers (>3σ), n={len(outliers)}")
    ax.set_title("TSLA Daily Return Outliers (>3 Standard Deviations)")
    ax.set_ylabel("% Return")
    ax.legend()
    save_fig(fig, "04_tsla_outliers.png")
    outliers.to_frame("return").to_csv(f"{PROC_DIR}/tsla_outliers.csv")

    # Visualization 5: distribution of returns (histogram + KDE) per asset
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, col in enumerate(returns.columns):
        sns.histplot(returns[col] * 100, kde=True, ax=axes[i], color=sns.color_palette()[i])
        axes[i].set_title(f"{col} Daily Return Distribution")
        axes[i].set_xlabel("% Return")
    save_fig(fig, "05_return_distributions.png")

    # Stationarity tests: closing prices vs daily returns for each asset
    adf_results = []
    for col in prices.columns:
        adf_results.append(eda.adf_test(prices[col], label=f"{col}_price"))
        adf_results.append(eda.adf_test(returns[col], label=f"{col}_return"))
    adf_df = pd.DataFrame([{
        "series": r["label"],
        "adf_statistic": r["adf_statistic"],
        "p_value": r["p_value"],
        "verdict": r["verdict"],
    } for r in adf_results])
    adf_df.to_csv(f"{PROC_DIR}/adf_test_results.csv", index=False)
    print("\nADF stationarity test results:\n", adf_df.round(4).to_string(index=False))

    with open(f"{PROC_DIR}/task1_insights.json", "w") as f:
        json.dump({
            "tsla_total_return_pct": float(stats.loc["TSLA", "total_return_pct"]),
            "tsla_annualized_volatility_pct": float(stats.loc["TSLA", "annualized_volatility_pct"]),
            "tsla_sharpe_ratio": float(stats.loc["TSLA", "sharpe_ratio"]),
            "tsla_var_95_daily_pct": float(stats.loc["TSLA", "var_95_daily_pct"]),
            "n_outliers_tsla": int(len(outliers)),
            "prices_stationary": {row["series"]: row["verdict"] for _, row in adf_df.iterrows() if "price" in row["series"]},
            "returns_stationary": {row["series"]: row["verdict"] for _, row in adf_df.iterrows() if "return" in row["series"]},
        }, f, indent=2)

    return prices, returns, stats


# ═══════════════════════════════════════════════════════════════════════════
# TASK 2 — Build Time Series Forecasting Models
# ═══════════════════════════════════════════════════════════════════════════
def task2_models(prices: pd.DataFrame):
    print("\n=== TASK 2: Build Time Series Forecasting Models ===")
    tsla = prices["TSLA"]
    train = tsla[:TRAIN_END]
    test = tsla[TEST_START:]
    print(f"Train: {train.index.min().date()} to {train.index.max().date()} ({len(train)} obs)")
    print(f"Test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} obs)")

    # ---- ARIMA ----
    print("\nRunning auto_arima to find best (p,d,q)...")
    try:
        order, seasonal_order, auto_model = arima_model.find_best_order(train, seasonal=False)
        print(f"Best ARIMA order: {order}")
        arima_fitted = arima_model.fit_arima(train, order)
        arima_forecast, arima_lower, arima_upper = arima_model.forecast_arima(arima_fitted, steps=len(test))
    except Exception as exc:
        raise RuntimeError("Task 2 failed during ARIMA model selection, fitting, or forecasting") from exc

    arima_forecast.index = test.index
    arima_lower.index = test.index
    arima_upper.index = test.index

    arima_mae = float(np.mean(np.abs(arima_forecast.values - test.values)))
    arima_rmse = float(np.sqrt(np.mean((arima_forecast.values - test.values) ** 2)))
    arima_mape = float(np.mean(np.abs((arima_forecast.values - test.values) / test.values)) * 100)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index[-200:], train.values[-200:], label="Train (last 200d)", color="gray", linewidth=1)
    ax.plot(test.index, test.values, label="Actual (Test)", color="black", linewidth=1.5)
    ax.plot(arima_forecast.index, arima_forecast.values, label=f"ARIMA{order} Forecast", color="crimson", linewidth=1.5)
    ax.fill_between(test.index, arima_lower.values, arima_upper.values, color="crimson", alpha=0.15, label="95% CI")
    ax.set_title(f"ARIMA{order} — TSLA Test Period Forecast vs Actual")
    ax.legend()
    save_fig(fig, "06_arima_test_forecast.png")

    # ---- LSTM ----
    print("\nTraining LSTM model...")
    window = 60
    train_vals = train.values.astype("float32")
    test_vals = test.values.astype("float32")
    lookback_vals = np.concatenate([train_vals[-window:], test_vals])

    try:
        train_scaled, lookback_scaled, scaler = lstm_model.train_test_scale(train_vals, lookback_vals)
        X_train, y_train = lstm_model.make_sequences(train_scaled, window=window)
        X_test, y_test = lstm_model.make_sequences(lookback_scaled, window=window)
        model = lstm_model.build_lstm_model(window=window, units=32, dropout=0.2, learning_rate=0.001)
        history = lstm_model.fit(model, X_train, y_train, epochs=10, batch_size=64,
                                  learning_rate=0.001, validation_split=0.1)
        lstm_pred_scaled = lstm_model.predict(model, X_test)
        lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
    except Exception as exc:
        raise RuntimeError("Task 2 failed during LSTM preprocessing, fitting, or inference") from exc

    lstm_mae = float(np.mean(np.abs(lstm_pred - test_vals)))
    lstm_rmse = float(np.sqrt(np.mean((lstm_pred - test_vals) ** 2)))
    lstm_mape = float(np.mean(np.abs((lstm_pred - test_vals) / test_vals)) * 100)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Val Loss")
    ax.set_title("LSTM Training Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled)")
    ax.legend()
    save_fig(fig, "07_lstm_training_loss.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index[-200:], train.values[-200:], label="Train (last 200d)", color="gray", linewidth=1)
    ax.plot(test.index, test_vals, label="Actual (Test)", color="black", linewidth=1.5)
    ax.plot(test.index, lstm_pred, label="LSTM Forecast", color="steelblue", linewidth=1.5)
    ax.set_title("LSTM — TSLA Test Period Forecast vs Actual")
    ax.legend()
    save_fig(fig, "08_lstm_test_forecast.png")

    comparison = pd.DataFrame([
        {"model": f"ARIMA{order}", "MAE": arima_mae, "RMSE": arima_rmse, "MAPE_pct": arima_mape},
        {"model": "LSTM(60d window)", "MAE": lstm_mae, "RMSE": lstm_rmse, "MAPE_pct": lstm_mape},
    ]).set_index("model")
    comparison.to_csv(f"{PROC_DIR}/model_comparison.csv")
    print("\nModel comparison:\n", comparison.round(3).to_string())

    best_model = comparison["RMSE"].idxmin()
    print(f"\nBest model by RMSE: {best_model}")

    import torch
    torch.save(model.state_dict(), f"{PROC_DIR}/lstm_model.pt")
    with open(f"{PROC_DIR}/arima_order.json", "w") as f:
        json.dump({"order": list(order)}, f)

    return {
        "order": order,
        "arima_fitted": arima_fitted,
        "arima_test_forecast": arima_forecast,
        "lstm_model": model,
        "lstm_scaler": scaler,
        "lstm_window": window,
        "comparison": comparison,
        "best_model": best_model,
        "train": train,
        "test": test,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TASK 3 — Forecast Future Market Trends
# ═══════════════════════════════════════════════════════════════════════════
def task3_forecast(prices: pd.DataFrame, task2_results: dict):
    print("\n=== TASK 3: Forecast Future Market Trends ===")
    tsla = prices["TSLA"]
    horizon_days = 252  # ~12 months of trading days
    last_date = tsla.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)

    order = task2_results["order"]
    try:
        full_model_fit = arima_model.fit_arima(tsla, order)
        forecast_mean, forecast_lower, forecast_upper = arima_model.forecast_arima(full_model_fit, steps=horizon_days)
    except Exception as exc:
        raise RuntimeError("Task 3 failed while fitting or forecasting the full-horizon ARIMA model") from exc

    forecast_mean.index = future_dates
    forecast_lower.index = future_dates
    forecast_upper.index = future_dates

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(tsla.index[-500:], tsla.values[-500:], label="Historical Close", color="black", linewidth=1.2)
    ax.plot(forecast_mean.index, forecast_mean.values, label=f"ARIMA{order} 12-Month Forecast", color="crimson", linewidth=1.6)
    ax.fill_between(forecast_mean.index, forecast_lower.values, forecast_upper.values,
                     color="crimson", alpha=0.18, label="95% Confidence Interval")
    ax.axvline(last_date, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"TSLA 12-Month Price Forecast with 95% Confidence Interval")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    save_fig(fig, "09_tsla_future_forecast.png")

    ci_width = forecast_upper - forecast_lower
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(forecast_mean.index, ci_width.values, color="darkorange", linewidth=1.6)
    ax.set_title("Widening of 95% Confidence Interval Over the Forecast Horizon")
    ax.set_ylabel("CI Width (USD)")
    ax.set_xlabel("Forecast Date")
    save_fig(fig, "10_ci_width_over_horizon.png")

    total_expected_return = (forecast_mean.iloc[-1] / tsla.iloc[-1] - 1)
    annualized_forecast_return = (1 + total_expected_return) ** (252 / horizon_days) - 1

    summary = {
        "last_actual_price": float(tsla.iloc[-1]),
        "forecast_horizon_days": horizon_days,
        "forecast_end_price_mean": float(forecast_mean.iloc[-1]),
        "forecast_end_price_lower_95": float(forecast_lower.iloc[-1]),
        "forecast_end_price_upper_95": float(forecast_upper.iloc[-1]),
        "total_expected_return_pct": float(total_expected_return * 100),
        "annualized_forecast_return_pct": float(annualized_forecast_return * 100),
        "ci_width_day1": float(ci_width.iloc[0]),
        "ci_width_final": float(ci_width.iloc[-1]),
        "ci_widening_factor": float(ci_width.iloc[-1] / ci_width.iloc[0]),
    }
    with open(f"{PROC_DIR}/task3_forecast_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    forecast_mean.to_frame("forecast").assign(lower=forecast_lower, upper=forecast_upper).to_csv(
        f"{PROC_DIR}/tsla_future_forecast.csv"
    )

    return summary, annualized_forecast_return


# ═══════════════════════════════════════════════════════════════════════════
# TASK 4 — Optimize Portfolio Based on Forecast
# ═══════════════════════════════════════════════════════════════════════════
def task4_portfolio(returns: pd.DataFrame, tsla_forecast_annual_return: float):
    print("\n=== TASK 4: Optimize Portfolio Based on Forecast ===")
    try:
        mu = portfolio.build_expected_returns(returns, tsla_forecast_annual_return)
        cov = portfolio.build_covariance_matrix(returns)
        vols, rets, weights_list = portfolio.efficient_frontier_points(mu, cov, n_points=50)
        max_sharpe_w, max_sharpe_perf = portfolio.max_sharpe_portfolio(mu, cov)
        min_vol_w, min_vol_perf = portfolio.min_volatility_portfolio(mu, cov)
    except Exception as exc:
        raise RuntimeError("Task 4 failed while building or solving the portfolio optimization problem") from exc

    print("Expected annual returns (mu):\n", mu.round(4))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cov, annot=True, fmt=".4f", cmap="coolwarm", ax=ax, square=True, cbar_kws={"label": "Covariance"})
    ax.set_title("Annualized Covariance Matrix (TSLA / BND / SPY)")
    save_fig(fig, "11_covariance_heatmap.png")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(vols, rets, "b--", linewidth=1.5, label="Efficient Frontier")
    ax.scatter(max_sharpe_perf[1], max_sharpe_perf[0], marker="*", color="red", s=350,
               label=f"Max Sharpe Portfolio (SR={max_sharpe_perf[2]:.2f})", zorder=5)
    ax.scatter(min_vol_perf[1], min_vol_perf[0], marker="*", color="green", s=350,
               label=f"Min Volatility Portfolio (SR={min_vol_perf[2]:.2f})", zorder=5)

    individual_vols = np.sqrt(np.diag(cov))
    for asset in mu.index:
        ax.scatter(individual_vols[list(mu.index).index(asset)], mu[asset], marker="o", s=90, label=f"{asset} (single asset)")

    ax.set_xlabel("Annualized Volatility (Risk)")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("Efficient Frontier — TSLA / BND / SPY")
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, "12_efficient_frontier.png")

    recommendation = {
        "max_sharpe_weights": max_sharpe_w,
        "max_sharpe_expected_return_pct": max_sharpe_perf[0] * 100,
        "max_sharpe_volatility_pct": max_sharpe_perf[1] * 100,
        "max_sharpe_ratio": max_sharpe_perf[2],
        "min_vol_weights": min_vol_w,
        "min_vol_expected_return_pct": min_vol_perf[0] * 100,
        "min_vol_volatility_pct": min_vol_perf[1] * 100,
        "min_vol_sharpe_ratio": min_vol_perf[2],
        "recommended_portfolio": "max_sharpe",
        "recommended_weights": max_sharpe_w,
    }
    with open(f"{PROC_DIR}/task4_portfolio_recommendation.json", "w") as f:
        json.dump(recommendation, f, indent=2, default=float)
    print(json.dumps(recommendation, indent=2, default=float))

    return recommendation


# ═══════════════════════════════════════════════════════════════════════════
# TASK 5 — Strategy Backtesting
# ═══════════════════════════════════════════════════════════════════════════
def task5_backtest(returns: pd.DataFrame, recommendation: dict):
    print("\n=== TASK 5: Strategy Backtesting ===")
    backtest_returns = returns[BACKTEST_START:]
    strategy_weights = recommendation["recommended_weights"]
    benchmark_weights = {"SPY": 0.6, "BND": 0.4, "TSLA": 0.0}

    strategy_returns = backtest.simulate_fixed_weights(backtest_returns, strategy_weights, rebalance=True)
    benchmark_returns = backtest.simulate_fixed_weights(backtest_returns, benchmark_weights, rebalance=True)

    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(strategy_cum.index, strategy_cum.values, label="Optimized Strategy Portfolio", linewidth=1.8, color="darkgreen")
    ax.plot(benchmark_cum.index, benchmark_cum.values, label="Benchmark (60% SPY / 40% BND)", linewidth=1.8, color="gray")
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_title("Backtest: Cumulative Returns — Strategy vs. Benchmark")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    save_fig(fig, "13_backtest_cumulative_returns.png")

    strategy_metrics = backtest.performance_metrics(strategy_returns)
    benchmark_metrics = backtest.performance_metrics(benchmark_returns)

    metrics_df = pd.DataFrame([strategy_metrics, benchmark_metrics], index=["Strategy", "Benchmark (60/40)"])
    metrics_df.to_csv(f"{PROC_DIR}/backtest_metrics.csv")
    print("\nBacktest performance metrics:\n", metrics_df.round(3).to_string())

    outperformed = strategy_metrics["total_return_pct"] > benchmark_metrics["total_return_pct"]
    with open(f"{PROC_DIR}/task5_backtest_conclusion.json", "w") as f:
        json.dump({
            "strategy_metrics": strategy_metrics,
            "benchmark_metrics": benchmark_metrics,
            "strategy_outperformed_benchmark": bool(outperformed),
        }, f, indent=2)

    return metrics_df, outperformed


def main(only=None):
    try:
        prices, returns, stats = task1_eda()
        if only == "1":
            return 0
        task2_results = task2_models(prices)
        if only == "2":
            return 0
        forecast_summary, tsla_annual_return = task3_forecast(prices, task2_results)
        if only == "3":
            return 0
        recommendation = task4_portfolio(returns, tsla_annual_return)
        if only == "4":
            return 0
        metrics_df, outperformed = task5_backtest(returns, recommendation)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE — all artifacts saved to data/processed/ and reports/figures/")
        print("=" * 60)
        return 0
    except Exception:
        logger.exception("Pipeline execution failed")
        return 1


if __name__ == "__main__":
    only_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(only=only_arg))
