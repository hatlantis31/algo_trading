"""
run_live_rebalance.py — once-a-month rebalancer for MarketBeatingStrategy on IBKR.

SAFE BY DEFAULT: runs in dry-run mode and connects to the paper port (7497).
Nothing is traded unless you explicitly pass --execute.

Usage:
    # ── Paper / dry-run (safe, always start here) ──────────────────────
    python scripts/run_live_rebalance.py

    # ── Paper + actually place orders on paper account ──────────────────
    python scripts/run_live_rebalance.py --execute

    # ── Live account (7496) — DO NOT RUN until you're confident ──────────
    python scripts/run_live_rebalance.py --execute --port 7496

Options:
    --execute           Place orders (default: dry-run, prints only)
    --port              IBKR TWS/Gateway port (default: 7497 paper)
    --capital AMOUNT    Override NAV, e.g. 50000 (default: full account NAV)
    --universe-size N   Number of stocks to include (default: 100; up to 589)
                        Larger = slower data fetch (~1.5 s per symbol)
    --min-trade AMOUNT  Skip orders below this USD value (default: 200)
    --order-type TYPE   MOC = Market On Close (default), MKT = immediate market

When to run:
    On the LAST TRADING DAY of each month, before 3:45 PM Eastern Time
    (NYSE MOC cut-off).  MOC orders fill at the official closing price.

    Tip: schedule with cron:
        # last business day of each month at 20:45 UTC (3:45 PM ET + buffer)
        45 20 28-31 * * [ "$(date +\\%u)" -le 5 ] && python /path/scripts/run_live_rebalance.py --execute
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_universe(size: int) -> list:
    """Load the live trading universe.

    Preference order:
    1. Tickers from the committed price parquet (previously validated liquid names)
    2. Fallback: S&P 500 tickers from sp500_loader (if parquet absent)

    Sorted by median dollar volume (best proxy for liquidity in the parquet).
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np

    parquet = Path("data/price_panel_daily.parquet")
    vol_parquet = Path("data/volume_panel_daily.parquet")

    if parquet.exists():
        prices = pd.read_parquet(parquet).astype(float)
        if vol_parquet.exists():
            volume = pd.read_parquet(vol_parquet).astype(float)
            dollar_vol = (prices * volume).median()
            tickers = dollar_vol.nlargest(size).index.tolist()
        else:
            coverage = prices.notna().sum()
            tickers = coverage.nlargest(size).index.tolist()
        logger.info("Universe: %d tickers from committed parquet", len(tickers))
        return tickers

    # Fallback: use the S&P 500 loader universe
    try:
        from core.sp500_loader import liquid_universe
        tickers = liquid_universe(size)
        logger.info("Universe: %d tickers from sp500_loader fallback", len(tickers))
        return tickers
    except Exception:
        pass

    raise SystemExit(
        "No universe available. Run scripts/export_price_panel.py first, or "
        "ensure data/price_panel_daily.parquet exists."
    )


def main():
    p = argparse.ArgumentParser(description="Monthly IBKR rebalancer for MarketBeatingStrategy")
    p.add_argument("--execute", action="store_true",
                   help="Actually place orders (default: dry-run only)")
    p.add_argument("--port", type=int, default=4002,
                   help="Gateway/TWS port: 4002 paper Gateway (default), "
                        "4001 live Gateway, 7497 TWS paper, 7496 TWS live")
    p.add_argument("--capital", type=float, default=None,
                   help="Override NAV in USD (default: full account NAV)")
    p.add_argument("--universe-size", type=int, default=100,
                   help="Number of stocks in the live universe (default 100)")
    p.add_argument("--min-trade", type=float, default=200.0,
                   help="Skip orders below this USD value (default 200)")
    p.add_argument("--order-type", default="MOC",
                   choices=["MOC", "MKT", "LMT"],
                   help="Order type: MOC (default), MKT, or LMT")
    args = p.parse_args()

    dry_run = not args.execute

    # ── Safety gates ─────────────────────────────────────────────────────────
    if args.port == 7496 and dry_run:
        logger.warning(
            "You specified live port 7496 but --execute was NOT passed. "
            "Running in dry-run mode (safe)."
        )
    if args.port == 7496 and args.execute:
        confirm = input(
            "\n⚠  LIVE PORT 7496 + --execute: this will place REAL orders "
            "on your live IBKR account.\n"
            "   Type 'YES I UNDERSTAND' to continue: "
        ).strip()
        if confirm != "YES I UNDERSTAND":
            print("Aborted.")
            sys.exit(0)

    # ── Connect to IBKR ───────────────────────────────────────────────────────
    try:
        from core.ibkr_connection import IBKRConnection
    except RuntimeError as e:
        raise SystemExit(f"ibapi not installed: {e}")

    conn = IBKRConnection(port=args.port)
    logger.info("Connecting to IBKR at port %d…", args.port)
    if not conn.connect():
        raise SystemExit(
            f"Could not connect to IBKR on port {args.port}. "
            "Make sure TWS or IB Gateway is running with the API enabled."
        )

    # ── Load universe ─────────────────────────────────────────────────────────
    universe = load_universe(args.universe_size)
    logger.info("Universe: %d symbols  (est. data fetch: %.0f s)",
                len(universe), len(universe) * 1.5)

    # ── Run rebalance ─────────────────────────────────────────────────────────
    from live.executor import LiveRebalancer

    rebalancer = LiveRebalancer(
        connection=conn,
        universe=universe,
        capital=args.capital,
        min_trade_value=args.min_trade,
        order_type=args.order_type,
    )

    try:
        orders = rebalancer.run(dry_run=dry_run)
        if not orders.empty:
            csv_path = Path("data") / f"orders_{__import__('datetime').date.today()}.csv"
            orders.to_csv(csv_path, index=False)
            logger.info("Order list saved to %s", csv_path)
    finally:
        conn.disconnect()

    if dry_run:
        print("\nDRY-RUN complete. Pass --execute to send these orders.")
    else:
        n_placed = len(orders)
        print(f"\n{n_placed} orders placed on IBKR ({args.order_type}, port {args.port}).")


if __name__ == "__main__":
    main()
