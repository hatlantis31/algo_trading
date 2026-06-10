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
│   ├── 07_sota_comparison.ipynb         # naive vs professional quant (real data)
│   ├── 09_hedge_fund_strategy.ipynb     # full market-neutral pipeline
│   └── 10_fundamental_data.ipynb        # adding fundamental value/quality factors
└── tests/               # pytest unit tests
```

## Tier 4 — market-beating long-only (beats buy-and-hold)

`strategies/long_only_factor.py` — `MarketBeatingStrategy`: a long-only strategy
that beats an equal-weight S&P 500 benchmark on return, Sharpe, and max drawdown.

**Validated on real point-in-time data (2020–2025, `scripts/validate_market_beating.py`)**

| strategy | Ann Return | Sharpe | Max DD | alpha/yr | IR | turnover |
|---|---|---|---|---|---|---|
| Equal-weight benchmark | +18.0% | 0.95 | −24.9% | — | — | — |
| v1: raw momentum, top 20% | +19.5% | 0.98 | −23.8% | +1.5% | 0.14 | 0.28 |
| **v2 (current): risk-adj momentum + rank buffer** | **+20.5%** | **1.09** | **−19.0%** | **+2.5%** | **0.25** | **0.16** |

Design levers (each tested in `scripts/research_improvements.py` — 11 variants head-to-head):
1. **Momentum selection** — hold only the top 20% of the universe ranked by a composite
   of 6-month momentum, 12-month momentum, 52-week-high proximity, and trend quality.
   No low-volatility or reversal factors — those bias toward low-beta defensive names
   that lag in bull markets.
2. **Risk-adjusted momentum** (Barroso & Santa-Clara 2015) — rank by momentum ÷ realized
   vol instead of raw momentum. Biggest single improvement: Sharpe 0.98 → 1.08, max DD
   −24% → −19%. Prefers smooth trends over volatile spikes.
3. **Sell-rank buffer** (Jegadeesh-Titman implementation practice) — buy at top-k, hold
   until rank falls below 2k. Halves turnover (0.28 → 0.16) and adds ~1% alpha by not
   churning positions that wobble around the selection boundary.
4. **Warm-up fallback** — equal-weight all stocks until 147 days of price history
   accumulate. Avoids dead-cash during the initial period.

Tested and **rejected** (kept honest — these did not help):
- regime filter (200-day MA): momentum already self-selects defensive names in bears
- quality tilt (PIT earnings yield): −0.04 Sharpe vs winner
- vol-targeted exposure scaling: no improvement over risk-adjusting the ranking itself
- tighter concentration (top 10/15%): more alpha but worse drawdown — noise-level Sharpe gain
- score-proportional weighting: +1.8% return but −0.02 Sharpe and +8.7% worse DD

Why it works year by year:
- **2022 (bear): −5.3%** vs benchmark −12.9% — risk-adjusted momentum holds the smooth
  2021 winners (energy, staples), not the volatile tech names that crashed.
- **2024 (AI bull): +32.3%** vs benchmark +17.6% — NVDA/META-style names have strong
  12-month momentum; the concentrated selection overweights them at ~1% each vs 0.17%
  in the equal-weight benchmark.

```python
from strategies.long_only_factor import MarketBeatingStrategy
from backtesting import PortfolioEngine
strat = MarketBeatingStrategy(volume_panel=vol)   # v2 defaults: risk-adj + buffer
result = PortfolioEngine("ME", cost_bps=10).run(close_panel, strat.weights)
```

> **Honest caveat**: +2.5% annual alpha, IR 0.25, t-stat 2.4 on a single 5-year sample;
> Deflated Sharpe 0.69 after counting all 21 research trials. The CPCV is solid (93%
> paths positive, mean Sharpe 1.13), but ~60 months cannot statistically separate a
> 2.5% alpha from luck. Real validation needs a longer out-of-sample window or a live
> forward test.

---

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

### Validated on true point-in-time data (2020–2025, notebook 11)

With real SimFin data (publish-date-aligned fundamentals, survivorship-bias-free
daily prices incl. delisted names like SIVB, window includes the 2022 bear),
**net of 10 bps costs** at 1.0x gross, dollar-neutral:

| configuration | Sharpe | maxDD | CPCV paths > 0 |
|---|---|---|---|
| **full pipeline: equal 16-factor blend, monthly** | **+0.50** | −9.7% | **87%** |
| fundamentals-only, IC-weighted + value-spread conditioned, monthly | +0.54 | −9.4% | 53% |
| full pipeline, IC-weighted, monthly | +0.09 | −18% | 53% |
| anything rebalanced weekly | ≤ 0 | up to −41% | — |

What the real data taught us (each one a classic result, reproduced honestly):
- **Slow beats fast**: weekly rebalancing quadruples costs and chases noise —
  monthly wins at every configuration (matches `core/diagnostics.py`).
- **Simple beats clever at high dimension**: equal-weighting 16 factors beats
  IC-weighting them (16 factors × 12 IC samples = estimation noise; DeMiguel 2009).
  IC-weighting only wins within the small correlated value-factor set.
- **Honest stats**: t ≈ 1.1, Deflated Sharpe ≈ 0.27 on ~5 years — a real but
  not-yet-significant edge; no 5-year sample of a 0.5-Sharpe strategy can be
  (see `live_feedback_horizon`).
- Real data also exposed two silent bugs synthetic data never caught: a NaN-collapse
  in the risk model and a circuit-breaker deadlock (a flat book can never recover
  its drawdown — fixed with a cooldown).

Run it yourself: `python scripts/validate_full_pipeline.py` — automatically runs the
full daily pipeline (price factors + point-in-time fundamentals + monthly rebalancing
+ risk overlay + CPCV + DSR) when `data/price_panel_daily.parquet` exists, else the
monthly fundamentals validation.

## Better-quality input — fundamental data (notebook 10)

Price/volume factors top out at IC ≈ 0.01–0.03. **Fundamental** value/quality
factors add *orthogonal* alpha price history cannot contain.

`core/fundamentals_loader.py` pulls a real S&P 500 fundamentals snapshot from
GitHub (**no API key, no registration** — works in any locked-down environment):
earnings yield, book-to-price, sales yield, EBITDA yield, dividend yield, size.

```python
from core.fundamentals_loader import download_fundamentals, fundamental_factors
download_fundamentals()                         # ~70 KB into data/ (gitignored)
fund = fundamental_factors(list(close.columns)) # static per-ticker value/quality frame
hf = HedgeFundStrategy(volume_panel=vol, fundamentals=fund,
                       alpha_combination="ic_weighted")  # IC-weighter learns each sign
```

> **Honest finding:** every value factor has *negative* IC in 2013–2018 ("value's
> lost decade"). Equal-weighting them is a disaster (−0.9 Sharpe); IC-weighting
> learns the sign and flips them into a net contributor (−0.09 → +0.11 Sharpe).
> Fundamental data helps **only when combined adaptively**.
>
> The bundled snapshot is dated ≈ Feb 2018 (the *end* of the price panel), so it is
> **look-ahead biased** for historical backtests. For clean point-in-time history
> run `scripts/ingest_fundamentals_local.py` on your own machine (SimFin free tier
> or yfinance — registration steps inside). At a <€10 budget the SimFin **free**
> tier is the right choice; no reputable sub-€10 point-in-time feed is worth buying.

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
