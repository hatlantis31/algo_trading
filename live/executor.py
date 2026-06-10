"""
LiveRebalancer — monthly portfolio rebalancer for MarketBeatingStrategy on IBKR.

Workflow (runs once per month, e.g. at 3:30 PM ET on the last trading day):

  1. Build a 1-year daily price panel from IBKR historical data (with pacing).
  2. Run MarketBeatingStrategy to get target weight allocation.
  3. Read current positions and Net Liquidation Value from IBKR.
  4. Compute target share counts: shares = floor(weight × nav / price).
  5. Diff target vs current → BUY / SELL orders.
  6. Filter tiny orders (below min_trade_value).
  7. Place orders if execute=True; otherwise print and return the order list.

Order type: MOC (Market On Close) by default — standard for monthly portfolios.
Send the orders before the MOC cut-off (3:45 PM ET on the NYSE) on the last
trading day of the month.

Usage:
    from core.ibkr_connection import IBKRConnection
    from live.executor import LiveRebalancer

    conn = IBKRConnection(port=7497)  # paper
    conn.connect()

    rebalancer = LiveRebalancer(conn, universe=["AAPL", "MSFT", ...])
    orders = rebalancer.run(dry_run=True)   # default: print + return, no trades
    orders = rebalancer.run(dry_run=False)  # actually place MOC orders on IBKR
"""

import logging
import math
from datetime import date

import pandas as pd

from strategies.long_only_factor import MarketBeatingStrategy

logger = logging.getLogger(__name__)


class LiveRebalancer:
    def __init__(
        self,
        connection,
        universe: list,
        capital: float = None,       # if None, uses full account NAV
        top_q: float = 0.20,
        min_trade_value: float = 200.0,  # skip orders worth less than this (USD)
        order_type: str = "MOC",         # MOC = Market On Close, MKT = immediate
        hist_delay: float = 1.5,         # seconds between IBKR hist-data requests
    ):
        self.conn = connection
        self.universe = list(universe)
        self.capital = capital
        self.top_q = top_q
        self.min_trade_value = min_trade_value
        self.order_type = order_type
        self.hist_delay = hist_delay

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, dry_run: bool = True) -> pd.DataFrame:
        """
        Compute orders for the next monthly rebalance.

        dry_run=True  (default): log everything, return order DataFrame, NO trades.
        dry_run=False           : place orders on IBKR (paper or live, per the port
                                  the IBKRConnection was opened on).

        Returns a DataFrame with columns:
            ticker, action, quantity, current_shares, target_shares,
            approx_price, approx_value_usd
        """
        tag = "[DRY-RUN] " if dry_run else ""
        logger.info("%sLive rebalance starting — %s", tag, date.today())

        # ── 1. Fetch price panel ─────────────────────────────────────────
        logger.info("Fetching 1-year daily bars for %d symbols…", len(self.universe))
        prices = self.conn.bulk_historical(
            self.universe, currency="USD", duration="1 Y",
            delay=self.hist_delay,
        )
        if prices.empty or len(prices) < 100:
            raise RuntimeError(
                "Insufficient price data returned from IBKR. "
                "Check connection, market-data subscription, and universe."
            )
        prices = prices.astype(float)
        prices = prices.where(prices > 0)
        logger.info("Panel: %d days × %d tickers", *prices.shape)

        # ── 2. Build target weights ──────────────────────────────────────
        strat = MarketBeatingStrategy(top_q=self.top_q)
        target_w = strat.weights(prices, prices.index[-1])
        if target_w.empty:
            raise RuntimeError(
                "Strategy returned empty weights — not enough price history. "
                "Run again when at least 147 trading days of data are available."
            )
        logger.info("Target: %d positions, gross=%.2f", (target_w > 0).sum(), target_w.sum())

        # ── 3. Fetch NAV + current positions ─────────────────────────────
        nav = self.capital or self.conn.get_nav()
        current_pos = self.conn.get_positions()   # {symbol: shares}
        logger.info("NAV: %.2f  Current positions: %d stocks", nav, len(current_pos))

        # ── 4. Compute target share counts ───────────────────────────────
        last_px = prices.iloc[-1]
        orders = self._build_orders(target_w, nav, current_pos, last_px)

        # ── 5. Print order list ───────────────────────────────────────────
        self._print_orders(orders, tag)

        # ── 6. Place orders if live ──────────────────────────────────────
        if not dry_run:
            placed = 0
            for _, row in orders.iterrows():
                if row["quantity"] > 0:
                    self.conn.place_order(
                        row["ticker"], row["action"], int(row["quantity"]),
                        order_type=self.order_type, dry_run=False,
                    )
                    placed += 1
            logger.info("Placed %d orders via IBKR (%s)", placed, self.order_type)
        else:
            logger.info("DRY-RUN: no orders sent. Pass dry_run=False to trade.")

        return orders

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_orders(
        self,
        target_w: pd.Series,
        nav: float,
        current_pos: dict,
        last_px: pd.Series,
    ) -> pd.DataFrame:
        rows = []
        all_tickers = set(target_w[target_w > 0].index) | set(current_pos.keys())

        for ticker in sorted(all_tickers):
            px = last_px.get(ticker)
            if px is None or px <= 0 or not math.isfinite(px):
                continue

            target_shares = math.floor(target_w.get(ticker, 0.0) * nav / px)
            current_shares = current_pos.get(ticker, 0)
            diff = target_shares - current_shares

            if diff == 0:
                continue
            approx_value = abs(diff) * px
            if approx_value < self.min_trade_value:
                continue

            rows.append({
                "ticker": ticker,
                "action": "BUY" if diff > 0 else "SELL",
                "quantity": abs(diff),
                "current_shares": current_shares,
                "target_shares": target_shares,
                "approx_price": round(px, 2),
                "approx_value_usd": round(approx_value, 2),
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["ticker", "action", "quantity", "current_shares",
                     "target_shares", "approx_price", "approx_value_usd"]
        )

    @staticmethod
    def _print_orders(orders: pd.DataFrame, tag: str):
        if orders.empty:
            logger.info("%sNo orders required (portfolio already at target).", tag)
            return
        buys  = orders[orders["action"] == "BUY"]
        sells = orders[orders["action"] == "SELL"]
        total_buy  = buys["approx_value_usd"].sum()
        total_sell = sells["approx_value_usd"].sum()
        print(f"\n{tag}═══ REBALANCE ORDERS  ({date.today()}) ════════════════")
        print(f"  BUYs:  {len(buys):3d}  total ~${total_buy:,.0f}")
        print(f"  SELLs: {len(sells):3d}  total ~${total_sell:,.0f}")
        print(f"  Net cash flow: ~${total_buy - total_sell:+,.0f}\n")
        for _, r in orders.sort_values(["action", "ticker"]).iterrows():
            print(f"  {r['action']:<4} {r['ticker']:<8} {int(r['quantity']):>5} shares "
                  f"× ${r['approx_price']:>8.2f}  ≈ ${r['approx_value_usd']:>10,.0f}  "
                  f"(was {int(r['current_shares'])} → {int(r['target_shares'])})")
        print()
