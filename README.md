# Algo Trading – IBKR / Ireland

A self-contained Python workspace for strategy research, backtesting, and live trading via Interactive Brokers.

## Structure

```
algo_trading/
├── config/              # IBKR connection settings, global parameters
├── core/                # IBKR connection, data fetcher, regime detector, S&P500 loader
├── strategies/          # Single-asset strategies + cross-sectional factor strategies
├── backtesting/         # Backtest engines, metrics, param tuner, evaluation suite
├── strategy_matrix/     # StrategyMatrix: score all strategies × all tickers
├── data/                # Local data cache (gitignored — never committed)
├── notebooks/           # Jupyter notebooks (numbered workflow)
│   ├── 01_ibkr_connection_test.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_strategy_overview.ipynb
│   ├── 04_backtest_runner.ipynb
│   ├── 05_strategy_matrix.ipynb
│   ├── 06_regime_and_tuning.ipynb       # regime filter + grid search
│   └── 07_sota_comparison.ipynb         # naive vs professional quant (real data)
└── tests/               # pytest unit tests (39 tests)
```

## Tier 3 — hedge-fund-grade market-neutral (the full pipeline)

`strategies/hedge_fund_strategy.py` runs the complete institutional pipeline used
by a systematic equity market-neutral fund:

```
price/volume → alpha factors → combine (IC-weight / ML) → neutralize (β, sector)
             → risk model (Ledoit-Wolf cov + betas) → MV optimizer
             → dollar- & beta-neutral, position-capped book
             → vol targeting + circuit breakers → CPCV + Deflated Sharpe
```

Components:
- `strategies/alpha_factors.py` — 10 cross-sectional signals (momentum, reversal,
  vol, liquidity, volume) + cross-sectional standardize/winsorize/neutralize + IC-weighting
- `strategies/ml_alpha.py` — HistGradientBoosting cross-sectional ranker (Gu-Kelly-Xiu)
  with strict purged training (no look-ahead)
- `backtesting/risk_model.py` — Ledoit-Wolf shrinkage covariance + per-stock betas
- `backtesting/optimizer.py` — analytic dollar/beta-neutral mean-variance with
  robustness shrinkage (DeMiguel 2009) and hard position caps
- `backtesting/cpcv.py` — Combinatorial Purged Cross-Validation (López de Prado)

```python
from strategies.hedge_fund_strategy import HedgeFundStrategy
from backtesting import PortfolioEngine
hf = HedgeFundStrategy(volume_panel=vol, alpha_combination="equal", construction="decile")
result = PortfolioEngine("ME", cost_bps=10).run(close_panel, hf.weights)
```

> **Honest result** on the bundled S&P data (notebook 09): ~0.3 Sharpe net,
> beta ≈ 0, −7% max DD — a real but modest market-independent edge from price/volume
> alone. The ML ranker and raw mean-variance did NOT beat the simple robust
> construction (over-engineering hurt). Real funds need fundamental data, a larger
> universe, and multi-regime history to push higher.

## Two tiers of strategy

**Tier 1 — single-asset (time-series).** Each strategy decides buy/sell on one ticker
from its own price history. Simple, but on real data they *lose to buy-and-hold* (they
short in bull markets and sit in cash). Use `BacktestEngine`.

**Tier 2 — cross-sectional (portfolio).** Rank the whole universe each rebalance and hold
a diversified book of the best names. This is how professional equity quant works. Use
`PortfolioEngine` with strategies from `strategies/cross_sectional.py`:
`CrossSectionalMomentum`, `ShortTermReversal`, `LowVolatility`, `MultiFactor`.

Supports inverse-vol position sizing and volatility targeting:
```python
from backtesting import PortfolioEngine
from strategies.cross_sectional import MultiFactor
eng = PortfolioEngine(rebalance="ME", cost_bps=10, inverse_vol=True, vol_target=0.12)
result = eng.run(price_panel, MultiFactor(long_short=False).weights)
```

## Professional evaluation suite (`backtesting/evaluation.py`)

A single Sharpe number lies. These tools tell you if an edge is real:
- `information_coefficient` — does the signal actually rank stocks? (0.02–0.05 = good)
- `deflated_sharpe_ratio` — Sharpe adjusted for how many variants you tried (López de Prado)
- `probabilistic_sharpe_ratio`, `t_stat_of_returns`, `drawdown_duration`
- `full_evaluation(result, n_trials=N)` — bundles everything

## Real data without a broker

`core/sp500_loader.py` bundles a real 5-year daily dataset (505 S&P 500 stocks):
```python
from core.sp500_loader import download_sp500, get_panel, liquid_universe
download_sp500()                       # one-time ~29 MB into data/ (gitignored)
panel = get_panel(liquid_universe(100))  # wide close-price panel for cross-sectional work
```
> Note: this dataset is **survivorship-biased** and covers only a bull market (2013–2018).
> Treat results as illustrative, not a validated edge.

## Quick Start

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Start with **notebook 02** if you don't have TWS running yet – it uses yfinance for offline data.

## IBKR Setup (Ireland)

1. Install TWS or IB Gateway from interactivebrokers.ie
2. Enable API: `Edit → Global Configuration → API → Settings`
   - Check *Enable ActiveX and Socket Clients*
   - Paper port: **7497** | Live port: **7496**
3. Edit `config/settings.py` with your port/client-id
4. Run **notebook 01** to verify the connection

## Strategies

| Key | Strategy | Type |
|-----|----------|------|
| `ma_cross` | Dual SMA Crossover | Trend-following |
| `rsi` | RSI Overbought/Oversold | Mean-reversion |
| `bollinger` | Bollinger Bands | Mean-reversion |
| `momentum` | N-day Price Momentum | Trend-following |
| `mean_reversion` | Z-score Mean Reversion | Statistical |

## Strategy Matrix

The `StrategyMatrix` (notebook 05) runs all strategies against all tickers,
builds a metric heatmap, and ranks the best (strategy, ticker) combinations.

```python
from strategy_matrix import StrategyMatrix

matrix = StrategyMatrix(tickers=['AAPL', 'CRH.L', 'SHEL.L'])
matrix.run()
matrix.show('sharpe_ratio')          # heatmap
matrix.select_best('sharpe_ratio', top_n=5)
```

## Tests

```bash
pytest tests/
```

## Notes for Ireland

- Base currency: **EUR** (set in `config/settings.py`)
- Your IBKR entity is **IBKR Ireland (IBKRIE)** – regulated by CBI
- Exchange suggestions: Euronext Dublin, LSE, Euronext Paris/Amsterdam
- Market hours: 08:00–16:30 GMT (winter) / 09:00–17:30 BST (summer)
