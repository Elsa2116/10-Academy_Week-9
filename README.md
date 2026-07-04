# Time Series Forecasting for Portfolio Management Optimization

**Client:** Guide Me in Finance (GMF) Investments
**Assets:** TSLA (high-growth/high-risk), BND (bond ETF, stability), SPY (S&P 500 ETF, diversified moderate risk)
**Data window:** 2015-01-01 to 2026-06-30 (source: Yahoo Finance via `yfinance`)

This project builds an end-to-end quantitative workflow that forecasts Tesla's
price trajectory, then uses that forecast to construct and backtest an
optimized multi-asset portfolio using Modern Portfolio Theory (MPT).

## Project tasks

| Task | Description | Key outputs |
|------|-------------|-------------|
| 1 | Preprocess & explore the data | Cleaning report, stationarity (ADF) tests, volatility/outlier analysis, risk metrics (VaR, Sharpe) |
| 2 | Build forecasting models | ARIMA/SARIMA (`pmdarima` auto-search) vs. LSTM (PyTorch), compared on MAE/RMSE/MAPE |
| 3 | Forecast future market trends | 12-month TSLA forecast with 95% confidence intervals |
| 4 | Optimize portfolio based on forecast | Efficient Frontier via `PyPortfolioOpt`, Max Sharpe & Min Volatility portfolios |
| 5 | Strategy backtesting | Optimized portfolio vs. 60/40 SPY/BND benchmark over the most recent 12 months |

## Repository structure

```
portfolio-optimization/
├── data/
│   ├── raw/               # Cached raw yfinance CSVs (gitignored)
│   └── processed/         # Cleaned data, metrics, model outputs (CSV/JSON)
├── notebooks/              # One notebook per task, fully executed with outputs
│   ├── 1.0-eda.ipynb
│   ├── 2.0-forecasting-models.ipynb
│   ├── 3.0-future-forecast.ipynb
│   ├── 4.0-portfolio-optimization.ipynb
│   └── 5.0-backtesting.ipynb
├── src/                     # Reusable, importable pipeline code
│   ├── data_loader.py       # yfinance fetch + cleaning
│   ├── eda.py                # Stationarity tests, VaR, Sharpe, outliers
│   ├── arima_model.py        # ARIMA/SARIMA fit + forecast
│   ├── lstm_model.py         # PyTorch LSTM model + training loop
│   ├── portfolio.py           # Efficient Frontier / MPT optimization
│   └── backtest.py            # Strategy simulation + performance metrics
├── scripts/
│   └── run_pipeline.py       # Runs all 5 tasks end-to-end, regenerates every artifact
├── tests/
│   └── test_basic.py          # Unit tests for src/ modules (pytest)
├── reports/
│   └── figures/                # All 13 generated plots (PNG)
├── .github/workflows/unittests.yml
├── .vscode/settings.json
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

Regenerate every data file and figure from scratch:

```bash
python scripts/run_pipeline.py
```

Run a single task only (1-5), e.g. just the EDA stage:

```bash
python scripts/run_pipeline.py 1
```

Data fetched from Yahoo Finance is cached under `data/raw/` so re-runs are fast
and do not require network access after the first run.

## Running tests

```bash
pytest tests/ -v
```

## Notebooks

Each notebook in `notebooks/` corresponds to one task, imports the reusable
`src/` modules, and contains the full narrative: markdown explanations, code,
and rendered visualizations. Open them in Jupyter or VS Code to review the
analysis interactively.

## Key findings (summary)

- **TSLA** delivered a ~2,717% total return over the sample period but with
  ~56% annualized volatility — by far the riskiest of the three assets.
- Both **price series are non-stationary** (ADF fails to reject the unit-root
  null); **daily returns are stationary**, confirming the standard practice of
  modeling returns/differenced prices rather than raw prices.
- The **LSTM model outperformed ARIMA** on the TSLA hold-out test set across
  all three error metrics (MAE, RMSE, MAPE), reflecting its ability to
  capture nonlinear patterns that a linear ARIMA cannot.
- The Efficient Frontier optimization, informed by the model's forecast,
  is recompiled by `scripts/run_pipeline.py`, and the recommended (Max Sharpe)
  allocation is documented in `data/processed/task4_portfolio_recommendation.json`.
- The backtest compares the optimized portfolio's realized performance against
  a static 60% SPY / 40% BND benchmark over the most recent 12 months; results
  are in `data/processed/backtest_metrics.csv`.

See the full narrative and charts in the notebooks, and the consolidated
**Investment Memo** PDF for the client-facing writeup.

## Disclaimer

This analysis is for educational purposes as part of a portfolio management
case study. Forecasts of financial markets (especially individual equities
like TSLA) are inherently uncertain; confidence intervals widen substantially
over the forecast horizon (see Task 3), and past performance does not
guarantee future results. This is not investment advice.
