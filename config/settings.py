"""
Global settings. Edit IBKR_HOST/PORT for your TWS/Gateway setup.
Ireland account: base currency EUR, UTC+0 (winter) / UTC+1 (summer).
"""

# ── IBKR connection ──────────────────────────────────────────────────────────
IBKR_HOST = "127.0.0.1"
IBKR_TWS_PORT = 7497          # TWS paper trading port (live: 7496)
IBKR_GATEWAY_PORT = 4002      # IB Gateway paper (live: 4001)
IBKR_CLIENT_ID = 1

# ── Account / region ────────────────────────────────────────────────────────
BASE_CURRENCY = "EUR"
TIMEZONE = "Europe/Dublin"

# ── Default backtest parameters ─────────────────────────────────────────────
BACKTEST_INITIAL_CAPITAL = 10_000.0   # EUR
BACKTEST_COMMISSION = 0.001           # 0.1% per trade (IBKR tiered approx.)
BACKTEST_SLIPPAGE = 0.0005            # 0.05%

# ── Data ─────────────────────────────────────────────────────────────────────
DEFAULT_BAR_SIZE = "1 day"
DEFAULT_DURATION = "1 Y"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
