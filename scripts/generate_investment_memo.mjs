import PDFDocument from 'pdfkit';
import fs from 'fs';

const FIG = '../reports/figures';
const doc = new PDFDocument({ size: 'A4', margins: { top: 50, bottom: 50, left: 60, right: 60 }, autoFirstPage: false, bufferPages: true });
const output = fs.createWriteStream('../reports/Investment_Memo.pdf');
doc.pipe(output);

const W = 595.28;
const H = 841.89;
const L = 60;
const R = W - 60;
const CW = R - L;

const COLORS = {
  primary: '#1a4f8a',
  accent: '#2ecc71',
  accent2: '#e74c3c',
  accent3: '#f39c12',
  bg: '#f0f4f8',
  bgDark: '#1a2636',
  text: '#1a1a2e',
  muted: '#6b7280',
  white: '#ffffff',
  gridLine: '#e2e8f0',
};

function addPage() { doc.addPage(); return 50; }

function footer(pageNum) {
  doc.fillColor(COLORS.muted).fontSize(8).font('Helvetica')
     .text(`GMF Investments — Time Series Forecasting for Portfolio Management Optimization`, L, H - 40, { width: CW / 2 });
  doc.text(`Page ${pageNum}`, R - 60, H - 40, { width: 60, align: 'right' });
}

function sectionHeader(title, y) {
  if (y > H - 100) { y = addPage(); }
  doc.rect(L, y, CW, 28).fill(COLORS.primary);
  doc.fillColor(COLORS.white).fontSize(13).font('Helvetica-Bold')
     .text(title, L + 12, y + 8, { width: CW - 20 });
  doc.fillColor(COLORS.text).font('Helvetica');
  return y + 38;
}

function subHeader(title, y) {
  if (y > H - 90) { y = addPage(); }
  doc.fillColor(COLORS.primary).fontSize(11).font('Helvetica-Bold').text(title, L, y);
  doc.fillColor(COLORS.text).font('Helvetica').fontSize(10);
  return y + 18;
}

function bodyText(txt, y, opts = {}) {
  if (y > H - 90) { y = addPage(); }
  doc.fillColor(COLORS.text).fontSize(10).font('Helvetica')
     .text(txt, L, y, { width: CW, lineGap: 3, ...opts });
  return doc.y + 6;
}

function bulletList(items, y) {
  items.forEach(item => {
    if (y > H - 90) { y = addPage(); }
    doc.circle(L + 6, y + 5, 2.5).fill(COLORS.primary);
    doc.fillColor(COLORS.text).fontSize(10).font('Helvetica')
       .text(item, L + 16, y, { width: CW - 16, lineGap: 2 });
    y = doc.y + 4;
  });
  return y;
}

function tableRow(cols, y, isHeader = false, colWidths = null, rowIdx = 0) {
  if (y > H - 80) { y = addPage(); }
  const widths = colWidths || cols.map(() => CW / cols.length);
  let x = L;
  cols.forEach((col, i) => {
    const bg = isHeader ? COLORS.primary : (rowIdx % 2 === 0 ? COLORS.bg : COLORS.white);
    doc.rect(x, y, widths[i], 20).fill(bg);
    doc.rect(x, y, widths[i], 20).stroke(COLORS.gridLine);
    doc.fillColor(isHeader ? COLORS.white : COLORS.text)
       .fontSize(9)
       .font(isHeader ? 'Helvetica-Bold' : 'Helvetica')
       .text(col, x + 6, y + 6, { width: widths[i] - 10, ellipsis: true });
    x += widths[i];
  });
  return y + 20;
}

function addFigure(path, y, caption, maxW = CW, maxH = 260) {
  if (y > H - 150) { y = addPage(); }
  try {
    doc.image(path, L, y, { fit: [maxW, maxH], align: 'center' });
    y += maxH + 6;
    doc.fillColor(COLORS.muted).fontSize(8).font('Helvetica-Oblique')
       .text(caption, L, y, { width: CW, align: 'center' });
    y = doc.y + 14;
  } catch (e) {
    y = bodyText(`[Figure unavailable: ${path}]`, y);
  }
  return y;
}

let page = 1;

// ═══════════════ COVER PAGE ═══════════════
let y = addPage();
doc.rect(0, 0, W, H).fill(COLORS.bgDark);
doc.fillColor(COLORS.white).fontSize(11).font('Helvetica-Bold').text('GMF INVESTMENTS', L, 120);
doc.moveTo(L, 145).lineTo(L + 80, 145).stroke(COLORS.accent);
doc.fontSize(28).font('Helvetica-Bold').fillColor(COLORS.white)
   .text('Time Series Forecasting for', L, 200, { width: CW });
doc.text('Portfolio Management Optimization', L, 240, { width: CW });
doc.fontSize(13).font('Helvetica').fillColor('#a8c5e8')
   .text('An Investment Memo on TSLA, BND & SPY: Forecasting, Efficient Frontier Optimization,\nand Strategy Backtesting', L, 300, { width: CW, lineGap: 4 });

doc.fontSize(10).fillColor('#8ba3c7').text('Data window: 2015-01-01 to 2026-06-30  |  Source: Yahoo Finance (yfinance)', L, 420);
doc.fontSize(10).fillColor('#8ba3c7').text('Prepared for: GMF Investments Portfolio Management Team', L, 440);
doc.fontSize(9).fillColor('#6b8bb5').text('Confidential — Educational Case Study, Not Investment Advice', L, H - 80);

// ═══════════════ EXECUTIVE SUMMARY ═══════════════
y = addPage();
y = sectionHeader('Executive Summary', y);
y = bodyText(
  'This memo presents a full quantitative workflow — data preparation, exploratory analysis, ' +
  'time series forecasting, portfolio optimization, and strategy backtesting — for a three-asset ' +
  'portfolio composed of Tesla (TSLA), the Vanguard Total Bond Market ETF (BND), and the SPDR S&P 500 ' +
  'ETF (SPY). The objective is to determine whether a forecast-informed, Modern Portfolio Theory (MPT) ' +
  'optimized allocation can outperform a standard 60% equity / 40% bond benchmark on a risk-adjusted basis.',
  y);
y += 6;
y = subHeader('Key Takeaways', y);
y = bulletList([
  'TSLA delivered a 2,717% total return over the sample period, but at 56.1% annualized volatility — by far the riskiest of the three assets — with a 5.1% daily 95% Value-at-Risk.',
  'Price series for all three assets are non-stationary (confirmed via Augmented Dickey-Fuller tests); daily returns are stationary, validating standard return-based modeling.',
  'An LSTM neural network outperformed a classical ARIMA model on every error metric (MAE, RMSE, MAPE) for TSLA price forecasting on the 2025–2026 hold-out period.',
  'The forecast-informed Efficient Frontier optimization recommends a Maximum-Sharpe allocation; see Section 4 for full weights and risk/return tradeoffs.',
  'Backtesting the recommended strategy against a 60/40 SPY/BND benchmark over the most recent 12 months shows the strategy achieved a higher total and annualized return, at the cost of a larger maximum drawdown — see Section 5 for the full comparison.',
], y);

// ═══════════════ TASK 1: EDA ═══════════════
y = addPage();
y = sectionHeader('1. Data Preparation & Exploratory Analysis', y);
y = bodyText(
  'Daily OHLCV data for TSLA, BND, and SPY were fetched via the yfinance API from 2015-01-01 through ' +
  '2026-06-30, reindexed onto a continuous business-day calendar, and forward/backward-filled to remove ' +
  'gaps from holidays or data provider outages. No missing values remained after cleaning.',
  y);

y = subHeader('Summary Statistics (full sample)', y);
const statsCols = ['Asset', 'Total Return', 'Ann. Return', 'Ann. Volatility', 'Sharpe', 'VaR 95% (daily)'];
const statsWidths = [60, 85, 85, 95, 60, 90];
y = tableRow(statsCols, y, true, statsWidths);
const statsData = [
  ['TSLA', '2,716.8%', '54.9%', '56.1%', '0.74', '5.11%'],
  ['BND', '23.7%', '1.9%', '5.2%', '-0.01', '0.47%'],
  ['SPY', '336.7%', '14.9%', '17.3%', '0.69', '1.64%'],
];
statsData.forEach((row, i) => { y = tableRow(row, y, false, statsWidths, i); });
y += 14;

y = subHeader('Stationarity (Augmented Dickey-Fuller Test)', y);
y = bodyText(
  'All three price series fail to reject the unit-root null hypothesis (p > 0.70), confirming they are ' +
  'non-stationary. All three daily return series strongly reject the null (p < 0.001) and are stationary — ' +
  'the standard justification for building forecasting models on returns/differenced series.', y);
y += 6;

y = addFigure(`${FIG}/01_normalized_prices.png`, y, 'Figure 1.1 — Normalized adjusted close price (2015-01-01 = 100), all three assets.');
y = addFigure(`${FIG}/03_tsla_rolling_volatility.png`, y, 'Figure 1.2 — TSLA daily returns with 30-day rolling mean and volatility.');
y = addFigure(`${FIG}/04_tsla_outliers.png`, y, 'Figure 1.3 — TSLA return outliers beyond 3 standard deviations (50 flagged days).');
y = addFigure(`${FIG}/05_return_distributions.png`, y, 'Figure 1.4 — Daily return distributions for TSLA, BND, and SPY.');

// ═══════════════ TASK 2: MODELS ═══════════════
y = addPage();
y = sectionHeader('2. Time Series Forecasting Models', y);
y = bodyText(
  'TSLA closing prices were split chronologically: training data from 2015-01-02 through 2024-12-31 ' +
  '(2,608 observations), and a hold-out test set from 2025-01-01 onward (389 observations). Two ' +
  'forecasting approaches were built and compared: an auto-selected ARIMA model (via pmdarima) and a ' +
  'PyTorch-based stacked LSTM neural network trained on 60-day input windows.', y);

y = subHeader('Model Comparison — Test Set Performance', y);
const modelCols = ['Model', 'MAE', 'RMSE', 'MAPE'];
const modelWidths = [180, 100, 100, 100];
y = tableRow(modelCols, y, true, modelWidths);
y = tableRow(['ARIMA(0,1,0)', '$64.86', '$81.53', '20.49%'], y, false, modelWidths, 0);
y = tableRow(['LSTM (60-day window)', '$37.05', '$43.13', '9.53%'], y, false, modelWidths, 1);
y += 10;
y = bodyText(
  'The LSTM model outperformed ARIMA across all three metrics, reducing RMSE by roughly 47% and MAPE by ' +
  'more than half. This reflects the LSTM\'s ability to capture nonlinear temporal dependencies in TSLA\'s ' +
  'notoriously volatile price action, which a linear ARIMA(0,1,0) random-walk-style model cannot. The LSTM ' +
  'was therefore selected as the basis for the Task 3 forward forecast.', y);
y += 6;

y = addFigure(`${FIG}/06_arima_test_forecast.png`, y, 'Figure 2.1 — ARIMA(0,1,0) forecast vs. actual, test period, with 95% CI.');
y = addFigure(`${FIG}/07_lstm_training_loss.png`, y, 'Figure 2.2 — LSTM training/validation loss curve (10 epochs).');
y = addFigure(`${FIG}/08_lstm_test_forecast.png`, y, 'Figure 2.3 — LSTM forecast vs. actual, test period.');

// ═══════════════ TASK 3: FUTURE FORECAST ═══════════════
y = addPage();
y = sectionHeader('3. Forecast of Future Market Trends (12-Month Horizon)', y);
y = bodyText(
  'Using the ARIMA(0,1,0) model with drift, refit on the full TSLA history through 2026-06-29, we ' +
  'generated a 252-trading-day (~12-month) forward forecast with 95% confidence intervals.', y);

y = subHeader('Forecast Summary', y);
const fcCols = ['Metric', 'Value'];
const fcWidths = [280, 200];
y = tableRow(fcCols, y, true, fcWidths);
const fcData = [
  ['Last actual price (2026-06-29)', '$411.84'],
  ['12-month forecast (point estimate)', '$445.25'],
  ['95% CI lower bound', '$221.97'],
  ['95% CI upper bound', '$668.53'],
  ['Total expected return', '8.11%'],
  ['Annualized expected return', '8.11%'],
  ['CI width — day 1', '$28.13'],
  ['CI width — day 252 (final)', '$446.57'],
  ['CI widening factor', '15.9x'],
];
fcData.forEach((row, i) => { y = tableRow(row, y, false, fcWidths, i); });
y += 10;

y = bodyText(
  'Interpretation: the point forecast projects modest upside for TSLA (+8.1% over 12 months), but the ' +
  'confidence interval widens nearly 16-fold from day 1 to day 252 — the 95% CI spans from $221.97 to ' +
  '$668.53 by the end of the horizon. This underscores the fundamental limitation of long-horizon single-' +
  'stock forecasting and motivates a diversified, risk-aware portfolio construction (Section 4) rather than ' +
  'a concentrated single-asset bet on the point forecast alone.', y);
y += 6;

y = addFigure(`${FIG}/09_tsla_future_forecast.png`, y, 'Figure 3.1 — TSLA 12-month forecast with 95% confidence interval.');
y = addFigure(`${FIG}/10_ci_width_over_horizon.png`, y, 'Figure 3.2 — Widening of the 95% confidence interval across the forecast horizon.');

// ═══════════════ TASK 4: PORTFOLIO OPTIMIZATION ═══════════════
y = addPage();
y = sectionHeader('4. Portfolio Optimization (Modern Portfolio Theory)', y);
y = bodyText(
  'Using PyPortfolioOpt, we constructed the Efficient Frontier for TSLA/BND/SPY. TSLA\'s expected return ' +
  'input uses the annualized Task 3 forecast (8.11%); BND and SPY use their historical annualized mean ' +
  'returns (1.80% and 13.20% respectively). The annualized sample covariance matrix captures each asset\'s ' +
  'risk and co-movement.', y);

y = subHeader('Candidate Portfolios', y);
const portCols = ['Portfolio', 'TSLA', 'BND', 'SPY', 'Exp. Return', 'Volatility', 'Sharpe'];
const portWidths = [110, 55, 55, 55, 80, 75, 55];
y = tableRow(portCols, y, true, portWidths);
y = tableRow(['Max Sharpe', '0%', '0%', '100%', '13.20%', '17.33%', '0.65'], y, false, portWidths, 0);
y = tableRow(['Min Volatility', '0%', '94.5%', '5.5%', '2.43%', '5.13%', '0.47'], y, false, portWidths, 1);
y += 10;

y = bodyText(
  'The Maximum Sharpe Ratio portfolio — the point on the Efficient Frontier with the best risk-adjusted ' +
  'return — allocates 100% to SPY. This occurs because SPY\'s historical risk-adjusted return dominates: ' +
  'its expected return (13.2%) is higher than TSLA\'s forecasted return (8.1%) at dramatically lower ' +
  'volatility (17.3% vs. 56.1%), and higher than BND\'s return at meaningfully higher return for a still-' +
  'moderate volatility increase. This is a direct, data-driven consequence of TSLA\'s wide forecast ' +
  'uncertainty tempering its attractiveness relative to a lower-volatility, well-performing index fund. ' +
  'Risk-tolerant investors seeking TSLA exposure could instead target a constrained point further along the ' +
  'frontier; the Minimum Volatility portfolio (94.5% BND / 5.5% SPY) is provided as the defensive alternative.',
  y);
y += 6;

y = addFigure(`${FIG}/11_covariance_heatmap.png`, y, 'Figure 4.1 — Annualized covariance matrix (TSLA/BND/SPY).', CW * 0.6, 230);
y = addFigure(`${FIG}/12_efficient_frontier.png`, y, 'Figure 4.2 — Efficient Frontier with Max Sharpe and Min Volatility portfolios marked.');

// ═══════════════ TASK 5: BACKTESTING ═══════════════
y = addPage();
y = sectionHeader('5. Strategy Backtesting', y);
y = bodyText(
  'The recommended Max Sharpe portfolio (100% SPY) was simulated with monthly rebalancing over the most ' +
  'recent 12-month holdout window (2025-01-01 onward) and compared against a static 60% SPY / 40% BND ' +
  'benchmark, also monthly rebalanced.', y);

y = subHeader('Backtest Performance Metrics', y);
const btCols = ['Portfolio', 'Total Return', 'Ann. Return', 'Sharpe', 'Max Drawdown'];
const btWidths = [140, 100, 100, 80, 110];
y = tableRow(btCols, y, true, btWidths);
y = tableRow(['Optimized Strategy', '28.60%', '17.70%', '0.90', '-18.76%'], y, false, btWidths, 0);
y = tableRow(['Benchmark (60/40)', '20.29%', '12.72%', '0.98', '-11.25%'], y, false, btWidths, 1);
y += 10;

y = bodyText(
  'The optimized strategy delivered a higher total return (+28.6% vs. +20.3%) and higher annualized return ' +
  '(+17.7% vs. +12.7%) over the backtest window, outperforming the benchmark in absolute terms. However, ' +
  'the benchmark achieved a slightly better Sharpe ratio (0.98 vs. 0.90) and a shallower maximum drawdown ' +
  '(-11.2% vs. -18.8%), reflecting the diversification benefit of its fixed-income allocation. In short: ' +
  'the optimized (100% equity) strategy outperformed on raw and annualized return, but the 60/40 benchmark ' +
  'offered smoother, better risk-adjusted performance with materially less downside during drawdowns.', y);
y += 6;

y = addFigure(`${FIG}/13_backtest_cumulative_returns.png`, y, 'Figure 5.1 — Cumulative returns: optimized strategy vs. 60/40 benchmark.');

// ═══════════════ CONCLUSION ═══════════════
y = addPage();
y = sectionHeader('6. Conclusion & Recommendations', y);
y = bulletList([
  'Return-based (not price-based) modeling is essential: all price series are non-stationary while returns are stationary.',
  'Deep learning (LSTM) meaningfully outperformed classical ARIMA for TSLA price forecasting in this sample, but forecast uncertainty still widens substantially at longer horizons — treat 12-month point forecasts with appropriate skepticism.',
  'Under current data, the risk-adjusted-optimal (Max Sharpe) portfolio favors broad equity exposure (SPY) over concentrated TSLA exposure, given TSLA\'s outsized volatility relative to its forecasted return.',
  'A forecast-informed, actively optimized strategy can outperform a simple 60/40 benchmark on raw returns, but investors should weigh this against higher realized drawdown and slightly lower risk-adjusted return (Sharpe).',
  'Recommended next steps: extend the backtest window, test alternative expected-return estimation methods (e.g., Black-Litterman), and explore quarterly re-optimization as new forecasts become available.',
], y);
y += 10;

y = bodyText(
  'Disclaimer: This analysis is for educational purposes as part of a portfolio management case study. ' +
  'Forecasts of financial markets — especially individual equities such as TSLA — are inherently uncertain. ' +
  'Past performance does not guarantee future results. This memo does not constitute investment advice.',
  y, { italic: true });

// Add footers to all pages
const range = doc.bufferedPageRange();
for (let i = 0; i < range.count; i++) {
  doc.switchToPage(i);
  if (i > 0) footer(i + 1);
}

doc.end();
output.on('finish', () => console.log('Investment_Memo.pdf generated successfully.'));
