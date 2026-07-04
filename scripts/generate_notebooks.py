"""
Generates the 5 task notebooks under notebooks/ using nbformat.
Each notebook imports the src/ modules and reproduces (a subset of) the
run_pipeline.py logic with markdown narrative cells interleaved for
readability. Run this, then execute each notebook with:

    jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
"""

import os

import nbformat as nbf

NB_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def write_notebook(filename, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    path = os.path.join(NB_DIR, filename)
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {path}")


SETUP = """import sys, os
sys.path.insert(0, os.path.abspath('..'))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.dpi'] = 110

from src import data_loader, eda, arima_model, lstm_model, portfolio, backtest
"""

# ── Notebook 1: EDA ──────────────────────────────────────────────────────
nb1 = [
    md("# Task 1 — Preprocess and Explore the Data\n\n"
       "Fetch historical data for **TSLA**, **BND**, and **SPY** (2015-01-01 to 2026-06-30) "
       "from Yahoo Finance, clean it, and explore trends, volatility, outliers, and stationarity."),
    code(SETUP),
    md("## 1.1 Fetch & clean data"),
    code("raw = data_loader.fetch_asset_data(cache_dir='../data/raw')\n"
         "cleaned = data_loader.clean_asset_data(raw)\n"
         "quality = data_loader.data_quality_report(raw, cleaned)\n"
         "quality"),
    md("## 1.2 Combine adjusted close prices and compute daily returns"),
    code("prices = data_loader.combine_asset_data(cleaned, field='Adj Close')\n"
         "returns = data_loader.compute_daily_returns(prices)\n"
         "prices.tail()"),
    md("## 1.3 Summary statistics & basic risk metrics (VaR, Sharpe)"),
    code("stats = eda.summary_statistics(prices, returns)\nstats.round(3)"),
    md("## 1.4 Visualize normalized closing prices"),
    code("normalized = prices / prices.iloc[0] * 100\n"
         "fig, ax = plt.subplots(figsize=(11, 5))\n"
         "normalized.plot(ax=ax, linewidth=1.6)\n"
         "ax.set_title('Normalized Adjusted Close Price (2015-01-01 = 100)')\n"
         "ax.set_ylabel('Indexed Price'); plt.show()"),
    md("## 1.5 Daily percentage change (volatility) by asset"),
    code("fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)\n"
         "for i, col in enumerate(returns.columns):\n"
         "    axes[i].plot(returns.index, returns[col]*100, linewidth=0.5)\n"
         "    axes[i].set_ylabel(f'{col}\\n% Change')\n"
         "plt.show()"),
    md("## 1.6 Rolling mean/volatility and outlier detection for TSLA"),
    code("roll_mean, roll_std = eda.rolling_stats(returns['TSLA'], window=30)\n"
         "outliers = eda.detect_outliers(returns['TSLA'], n_std=3.0)\n"
         "fig, ax = plt.subplots(figsize=(11, 5))\n"
         "ax.plot(returns.index, returns['TSLA']*100, alpha=0.3, linewidth=0.6, label='Daily Return')\n"
         "ax.plot(roll_mean.index, roll_mean*100, label='30D Rolling Mean')\n"
         "ax.plot(roll_std.index, roll_std*100, label='30D Rolling Std')\n"
         "ax.scatter(outliers.index, outliers*100, color='crimson', s=20, label=f'Outliers n={len(outliers)}')\n"
         "ax.legend(); plt.show()"),
    md("## 1.7 Stationarity testing (Augmented Dickey-Fuller)\n\n"
       "Test both raw prices and daily returns for each asset. We expect **prices to be "
       "non-stationary** (unit root, trending) and **returns to be stationary** — this "
       "justifies modeling returns/differenced series rather than raw prices."),
    code("results = []\n"
         "for col in prices.columns:\n"
         "    results.append(eda.adf_test(prices[col], label=f'{col}_price'))\n"
         "    results.append(eda.adf_test(returns[col], label=f'{col}_return'))\n"
         "adf_df = pd.DataFrame([{'series': r['label'], 'adf_stat': r['adf_statistic'], 'p_value': r['p_value'], 'verdict': r['verdict']} for r in results])\n"
         "adf_df"),
    md("### Key Insight\n\n"
       "All three price series fail to reject the unit-root null hypothesis (p > 0.05), "
       "confirming they are **non-stationary random walks with drift** — typical of "
       "financial asset prices. All three return series strongly reject the null "
       "(p < 0.001) and are **stationary**, which is why forecasting models in Task 2 "
       "are built on (differenced) price/return series rather than raw prices directly."),
]
write_notebook("1.0-eda.ipynb", nb1)

# ── Notebook 2: Forecasting models ───────────────────────────────────────
nb2 = [
    md("# Task 2 — Build Time Series Forecasting Models\n\n"
       "Compare a classical **ARIMA** model against a **PyTorch LSTM** neural network "
       "for forecasting TSLA's closing price, using a chronological train/test split "
       "(train through 2024-12-31, test on 2025-01-01 onward)."),
    code(SETUP),
    code("raw = data_loader.fetch_asset_data(cache_dir='../data/raw')\n"
         "cleaned = data_loader.clean_asset_data(raw)\n"
         "prices = data_loader.combine_asset_data(cleaned, field='Adj Close')\n"
         "tsla = prices['TSLA']\n"
         "train, test = tsla[:'2024-12-31'], tsla['2025-01-01':]\n"
         "print(f'Train: {len(train)} obs, Test: {len(test)} obs')"),
    md("## 2.1 ARIMA — automatic order selection"),
    code("order, seasonal_order, auto_model = arima_model.find_best_order(train, seasonal=False)\n"
         "print('Best order (p,d,q):', order)\n"
         "arima_fitted = arima_model.fit_arima(train, order)\n"
         "arima_fitted.summary()"),
    code("arima_fc, arima_lo, arima_hi = arima_model.forecast_arima(arima_fitted, steps=len(test))\n"
         "arima_fc.index = arima_lo.index = arima_hi.index = test.index\n"
         "fig, ax = plt.subplots(figsize=(11,5))\n"
         "ax.plot(test.index, test.values, label='Actual', color='black')\n"
         "ax.plot(arima_fc.index, arima_fc.values, label=f'ARIMA{order} Forecast', color='crimson')\n"
         "ax.fill_between(test.index, arima_lo, arima_hi, color='crimson', alpha=0.15, label='95% CI')\n"
         "ax.legend(); ax.set_title('ARIMA Forecast vs Actual (Test Period)'); plt.show()"),
    md("## 2.2 LSTM (PyTorch)"),
    code("window = 60\n"
         "train_vals = train.values.astype('float32')\n"
         "test_vals = test.values.astype('float32')\n"
         "lookback_vals = np.concatenate([train_vals[-window:], test_vals])\n"
         "train_scaled, lookback_scaled, scaler = lstm_model.train_test_scale(train_vals, lookback_vals)\n"
         "X_train, y_train = lstm_model.make_sequences(train_scaled, window=window)\n"
         "X_test, y_test = lstm_model.make_sequences(lookback_scaled, window=window)\n"
         "model = lstm_model.build_lstm_model(window=window, units=32, dropout=0.2)\n"
         "history = lstm_model.fit(model, X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)\n"
         "print('Final train loss:', history['loss'][-1], ' val loss:', history['val_loss'][-1])"),
    code("fig, ax = plt.subplots(figsize=(8,4))\n"
         "ax.plot(history['loss'], label='Train Loss'); ax.plot(history['val_loss'], label='Val Loss')\n"
         "ax.legend(); ax.set_title('LSTM Training Loss'); plt.show()"),
    code("lstm_pred_scaled = lstm_model.predict(model, X_test)\n"
         "lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()\n"
         "fig, ax = plt.subplots(figsize=(11,5))\n"
         "ax.plot(test.index, test_vals, label='Actual', color='black')\n"
         "ax.plot(test.index, lstm_pred, label='LSTM Forecast', color='steelblue')\n"
         "ax.legend(); ax.set_title('LSTM Forecast vs Actual (Test Period)'); plt.show()"),
    md("## 2.3 Model comparison: MAE, RMSE, MAPE"),
    code("def metrics(pred, actual):\n"
         "    pred, actual = np.asarray(pred), np.asarray(actual)\n"
         "    mae = np.mean(np.abs(pred - actual))\n"
         "    rmse = np.sqrt(np.mean((pred - actual)**2))\n"
         "    mape = np.mean(np.abs((pred - actual)/actual)) * 100\n"
         "    return mae, rmse, mape\n"
         "\n"
         "arima_metrics = metrics(arima_fc.values, test.values)\n"
         "lstm_metrics = metrics(lstm_pred, test_vals)\n"
         "comparison = pd.DataFrame([\n"
         "    {'model': f'ARIMA{order}', 'MAE': arima_metrics[0], 'RMSE': arima_metrics[1], 'MAPE_pct': arima_metrics[2]},\n"
         "    {'model': 'LSTM', 'MAE': lstm_metrics[0], 'RMSE': lstm_metrics[1], 'MAPE_pct': lstm_metrics[2]},\n"
         "]).set_index('model')\n"
         "comparison.round(3)"),
    md("### Conclusion\n\n"
       "The model with the lower RMSE/MAE/MAPE is selected as the basis for the Task 3 "
       "future forecast and the Task 4 expected-return input for TSLA."),
]
write_notebook("2.0-forecasting-models.ipynb", nb2)

# ── Notebook 3: Future forecast ──────────────────────────────────────────
nb3 = [
    md("# Task 3 — Forecast Future Market Trends\n\n"
       "Generate a 12-month (252 trading day) forecast for TSLA using the best model "
       "from Task 2, with 95% confidence intervals, and interpret trend + uncertainty."),
    code(SETUP),
    code("raw = data_loader.fetch_asset_data(cache_dir='../data/raw')\n"
         "cleaned = data_loader.clean_asset_data(raw)\n"
         "prices = data_loader.combine_asset_data(cleaned, field='Adj Close')\n"
         "tsla = prices['TSLA']\n"
         "import json\n"
         "order = tuple(json.load(open('../data/processed/arima_order.json'))['order'])\n"
         "order"),
    md("## 3.1 Fit on full history & forecast 12 months ahead"),
    code("horizon = 252\n"
         "full_fit = arima_model.fit_arima(tsla, order)\n"
         "fc_mean, fc_lo, fc_hi = arima_model.forecast_arima(full_fit, steps=horizon)\n"
         "future_dates = pd.bdate_range(start=tsla.index[-1] + pd.Timedelta(days=1), periods=horizon)\n"
         "fc_mean.index = fc_lo.index = fc_hi.index = future_dates\n"
         "fc_mean.tail()"),
    code("fig, ax = plt.subplots(figsize=(12,5.5))\n"
         "ax.plot(tsla.index[-500:], tsla.values[-500:], label='Historical', color='black')\n"
         "ax.plot(fc_mean.index, fc_mean.values, label='12-Month Forecast', color='crimson')\n"
         "ax.fill_between(fc_mean.index, fc_lo, fc_hi, color='crimson', alpha=0.18, label='95% CI')\n"
         "ax.legend(); ax.set_title('TSLA 12-Month Forecast'); plt.show()"),
    md("## 3.2 Confidence interval widening over the horizon\n\n"
       "As is expected for a differenced/random-walk-style model, the confidence interval "
       "widens over time, reflecting growing forecast uncertainty further into the future."),
    code("ci_width = fc_hi - fc_lo\n"
         "fig, ax = plt.subplots(figsize=(11,4.5))\n"
         "ax.plot(fc_mean.index, ci_width.values, color='darkorange')\n"
         "ax.set_title('95% CI Width Over the Forecast Horizon'); plt.show()\n"
         "print('CI width day 1:', round(ci_width.iloc[0],2), ' | CI width final day:', round(ci_width.iloc[-1],2))"),
    md("## 3.3 Interpretation\n\n"
       "- **Trend**: whether the point forecast is upward, downward, or flat depends on the "
       "fitted drift term of the chosen model — see the printed forecast summary above.\n"
       "- **Opportunities**: an upward trend with a still-reasonable lower bound suggests "
       "room for a modest TSLA allocation.\n"
       "- **Risks**: the wide and widening confidence interval highlights how unreliable "
       "point forecasts become at longer horizons — position sizing should stay conservative, "
       "and this uncertainty band is fed directly into the risk-aware Task 4 optimization."),
]
write_notebook("3.0-future-forecast.ipynb", nb3)

# ── Notebook 4: Portfolio optimization ───────────────────────────────────
nb4 = [
    md("# Task 4 — Optimize Portfolio Based on Forecast\n\n"
       "Use Modern Portfolio Theory (via `PyPortfolioOpt`) to build the Efficient Frontier "
       "for TSLA/BND/SPY, using the Task 3 TSLA forecast as its expected return input."),
    code(SETUP),
    code("import json\n"
         "raw = data_loader.fetch_asset_data(cache_dir='../data/raw')\n"
         "cleaned = data_loader.clean_asset_data(raw)\n"
         "prices = data_loader.combine_asset_data(cleaned, field='Adj Close')\n"
         "returns = data_loader.compute_daily_returns(prices)\n"
         "forecast_summary = json.load(open('../data/processed/task3_forecast_summary.json'))\n"
         "tsla_annual_return = forecast_summary['annualized_forecast_return_pct'] / 100\n"
         "tsla_annual_return"),
    md("## 4.1 Expected returns & covariance matrix"),
    code("mu = portfolio.build_expected_returns(returns, tsla_annual_return)\n"
         "cov = portfolio.build_covariance_matrix(returns)\n"
         "mu"),
    code("fig, ax = plt.subplots(figsize=(6,5))\n"
         "sns.heatmap(cov, annot=True, fmt='.4f', cmap='coolwarm', square=True, ax=ax)\n"
         "ax.set_title('Annualized Covariance Matrix'); plt.show()"),
    md("## 4.2 Efficient Frontier"),
    code("vols, rets, weights_list = portfolio.efficient_frontier_points(mu, cov, n_points=50)\n"
         "max_sharpe_w, max_sharpe_perf = portfolio.max_sharpe_portfolio(mu, cov)\n"
         "min_vol_w, min_vol_perf = portfolio.min_volatility_portfolio(mu, cov)\n"
         "print('Max Sharpe weights:', max_sharpe_w)\n"
         "print('Min Vol weights:', min_vol_w)"),
    code("fig, ax = plt.subplots(figsize=(9,6.5))\n"
         "ax.plot(vols, rets, 'b--', label='Efficient Frontier')\n"
         "ax.scatter(max_sharpe_perf[1], max_sharpe_perf[0], marker='*', color='red', s=300, label='Max Sharpe')\n"
         "ax.scatter(min_vol_perf[1], min_vol_perf[0], marker='*', color='green', s=300, label='Min Volatility')\n"
         "for asset in mu.index:\n"
         "    idx = list(mu.index).index(asset)\n"
         "    ax.scatter(np.sqrt(cov.values[idx, idx]), mu[asset], s=80, label=asset)\n"
         "ax.set_xlabel('Volatility'); ax.set_ylabel('Expected Return'); ax.legend(fontsize=8)\n"
         "ax.set_title('Efficient Frontier'); plt.show()"),
    md("## 4.3 Recommendation\n\n"
       "The **Maximum Sharpe Ratio portfolio** is recommended as it offers the best "
       "risk-adjusted return among all points on the Efficient Frontier. Its weights, "
       "expected return, volatility, and Sharpe ratio are printed above and saved to "
       "`data/processed/task4_portfolio_recommendation.json`."),
]
write_notebook("4.0-portfolio-optimization.ipynb", nb4)

# ── Notebook 5: Backtesting ──────────────────────────────────────────────
nb5 = [
    md("# Task 5 — Strategy Backtesting\n\n"
       "Backtest the Task 4 optimized portfolio against a static 60% SPY / 40% BND "
       "benchmark over the most recent 12-month holdout window (2025-01-01 onward)."),
    code(SETUP),
    code("import json\n"
         "raw = data_loader.fetch_asset_data(cache_dir='../data/raw')\n"
         "cleaned = data_loader.clean_asset_data(raw)\n"
         "prices = data_loader.combine_asset_data(cleaned, field='Adj Close')\n"
         "returns = data_loader.compute_daily_returns(prices)\n"
         "recommendation = json.load(open('../data/processed/task4_portfolio_recommendation.json'))\n"
         "strategy_weights = recommendation['recommended_weights']\n"
         "strategy_weights"),
    md("## 5.1 Simulate strategy vs. benchmark (monthly rebalanced)"),
    code("backtest_returns = returns['2025-01-01':]\n"
         "benchmark_weights = {'SPY': 0.6, 'BND': 0.4, 'TSLA': 0.0}\n"
         "strategy_returns = backtest.simulate_fixed_weights(backtest_returns, strategy_weights, rebalance=True)\n"
         "benchmark_returns = backtest.simulate_fixed_weights(backtest_returns, benchmark_weights, rebalance=True)\n"
         "strategy_cum = (1 + strategy_returns).cumprod()\n"
         "benchmark_cum = (1 + benchmark_returns).cumprod()\n"
         "fig, ax = plt.subplots(figsize=(11,5.5))\n"
         "ax.plot(strategy_cum.index, strategy_cum.values, label='Optimized Strategy', color='darkgreen')\n"
         "ax.plot(benchmark_cum.index, benchmark_cum.values, label='60/40 Benchmark', color='gray')\n"
         "ax.axhline(1.0, color='black', linewidth=0.6, linestyle=':')\n"
         "ax.legend(); ax.set_title('Backtest: Cumulative Returns'); plt.show()"),
    md("## 5.2 Performance metrics"),
    code("strategy_metrics = backtest.performance_metrics(strategy_returns)\n"
         "benchmark_metrics = backtest.performance_metrics(benchmark_returns)\n"
         "metrics_df = pd.DataFrame([strategy_metrics, benchmark_metrics], index=['Strategy', 'Benchmark (60/40)'])\n"
         "metrics_df.round(3)"),
    md("## 5.3 Conclusion\n\n"
       "Compare total return, annualized return, Sharpe ratio, and max drawdown between "
       "the two approaches (see table above). This determines whether the "
       "forecast-informed, MPT-optimized strategy outperformed the simple static "
       "benchmark over the backtest window, and at what cost in terms of volatility "
       "and drawdown risk."),
]
write_notebook("5.0-backtesting.ipynb", nb5)

print("\nAll 5 notebooks generated.")
