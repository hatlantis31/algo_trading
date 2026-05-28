# Algo Trading – IBKR / Ireland

A self-contained Python workspace for strategy research, backtesting, and live trading via Interactive Brokers.

## Structure

```
algo_trading/
├── config/              # IBKR connection settings, global parameters
├── core/                # IBKR connection wrapper, data fetcher
├── strategies/          # Strategy library (base class + 5 strategies)
├── backtesting/         # Vectorised backtest engine + performance metrics
├── strategy_matrix/     # StrategyMatrix: score all strategies × all tickers
├── notebooks/           # Jupyter notebooks (numbered workflow)
│   ├── 01_ibkr_connection_test.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_strategy_overview.ipynb
│   ├── 04_backtest_runner.ipynb
│   └── 05_strategy_matrix.ipynb
└── tests/               # pytest unit tests
```

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
