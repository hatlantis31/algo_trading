"""
Thin wrapper around ibapi to manage TWS / IB Gateway connections.

Usage:
    conn = IBKRConnection(port=7497)   # paper TWS
    conn.connect()
    df = conn.get_historical_data("AAPL", exchange="SMART", currency="USD")
    panel = conn.bulk_historical(["AAPL", "MSFT", "NVDA"])  # paced, returns wide panel
    nav = conn.get_nav()
    positions = conn.get_positions()                         # {symbol: shares}
    order_id = conn.place_order("AAPL", "BUY", 10)         # dry_run by default
    conn.disconnect()
"""

import time
import threading
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.common import BarData
    IBAPI_AVAILABLE = True
except ImportError:
    IBAPI_AVAILABLE = False
    logging.warning("ibapi not installed – IBKR live connection unavailable.")

from config.settings import IBKR_HOST, IBKR_TWS_PORT, IBKR_CLIENT_ID

logger = logging.getLogger(__name__)

# IBKR pacing rule: max ~50 historical-data requests per 10 minutes.
# 1.5 s between requests is conservative but safe for all subscription tiers.
HIST_REQUEST_DELAY_S = 1.5


if IBAPI_AVAILABLE:
    class _IBWrapper(EWrapper):
        def __init__(self):
            super().__init__()
            self._bars: list[BarData] = []
            self._hist_done = threading.Event()
            self._pos_done = threading.Event()
            self._acct_done = threading.Event()
            self.next_order_id: Optional[int] = None
            self.account_summary: dict = {}
            self._positions: dict = {}   # symbol → int shares
            self._order_statuses: dict = {}

        # ── Historical data ──────────────────────────────────────────────
        def historicalData(self, reqId, bar):
            self._bars.append(bar)

        def historicalDataEnd(self, reqId, start, end):
            self._hist_done.set()

        # ── Order management ─────────────────────────────────────────────
        def nextValidId(self, orderId):
            self.next_order_id = orderId

        def orderStatus(self, orderId, status, filled, remaining,
                        avgFillPrice, permId, parentId, lastFillPrice,
                        clientId, whyHeld, mktCapPrice):
            self._order_statuses[orderId] = {
                "status": status, "filled": filled,
                "avg_price": avgFillPrice,
            }
            logger.info("Order %s: %s filled=%s avgPx=%.2f",
                        orderId, status, filled, avgFillPrice)

        # ── Account & positions ──────────────────────────────────────────
        def accountSummary(self, reqId, account, tag, value, currency):
            self.account_summary[tag] = (value, currency)

        def accountSummaryEnd(self, reqId):
            self._acct_done.set()

        def position(self, account, contract, position, avgCost):
            if contract.secType == "STK" and position != 0:
                self._positions[contract.symbol] = int(position)

        def positionEnd(self):
            self._pos_done.set()

        def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
            if errorCode not in (2104, 2106, 2158, 2168):  # benign info codes
                logger.error("IBKR error %s (reqId %s): %s",
                             errorCode, reqId, errorString)
            # Signal completion for historical data errors (e.g. no data)
            if reqId == 1 and errorCode in (162, 321, 200):
                self._hist_done.set()

    class _IBClient(EClient):
        def __init__(self, wrapper):
            super().__init__(wrapper)

    class IBKRConnection:
        def __init__(self, host=IBKR_HOST, port=IBKR_TWS_PORT, client_id=IBKR_CLIENT_ID):
            self.host = host
            self.port = port
            self.client_id = client_id
            self._wrapper = _IBWrapper()
            self._client = _IBClient(self._wrapper)
            self._thread: Optional[threading.Thread] = None
            self._order_counter = 0

        def connect(self, timeout: float = 10.0) -> bool:
            self._client.connect(self.host, self.port, self.client_id)
            self._thread = threading.Thread(target=self._client.run, daemon=True)
            self._thread.start()
            deadline = time.time() + timeout
            while self._wrapper.next_order_id is None and time.time() < deadline:
                time.sleep(0.1)
            connected = self._wrapper.next_order_id is not None
            if connected:
                logger.info("Connected to IBKR at %s:%s (clientId=%s)",
                            self.host, self.port, self.client_id)
            else:
                logger.error("Failed to connect within %.1fs", timeout)
            return connected

        def disconnect(self):
            self._client.disconnect()
            logger.info("Disconnected from IBKR")

        # ── Historical data ───────────────────────────────────────────────

        def get_historical_data(
            self,
            symbol: str,
            exchange: str = "SMART",
            currency: str = "USD",
            sec_type: str = "STK",
            duration: str = "1 Y",
            bar_size: str = "1 day",
            what_to_show: str = "ADJUSTED_LAST",
        ) -> pd.DataFrame:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = sec_type
            contract.exchange = exchange
            contract.currency = currency

            self._wrapper._bars.clear()
            self._wrapper._hist_done.clear()
            req_id = 1
            end_dt = datetime.now().strftime("%Y%m%d %H:%M:%S")
            self._client.reqHistoricalData(
                req_id, contract, end_dt, duration, bar_size,
                what_to_show, 1, 1, False, []
            )
            self._wrapper._hist_done.wait(timeout=30)

            rows = [
                {"date": b.date, "open": b.open, "high": b.high,
                 "low": b.low, "close": b.close, "volume": b.volume}
                for b in self._wrapper._bars
            ]
            df = pd.DataFrame(rows)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            return df

        def bulk_historical(
            self,
            symbols: list,
            exchange: str = "SMART",
            currency: str = "USD",
            duration: str = "1 Y",
            bar_size: str = "1 day",
            delay: float = HIST_REQUEST_DELAY_S,
        ) -> pd.DataFrame:
            """Fetch 1y close price panel for a list of symbols, respecting IBKR
            pacing limits (default 1.5 s between requests).

            Returns a wide DataFrame: rows=dates, cols=symbol (close prices).
            Symbols that fail or return no data are silently skipped.
            """
            close_series = {}
            n = len(symbols)
            for i, sym in enumerate(symbols):
                if i > 0:
                    time.sleep(delay)
                logger.info("Fetching %s (%d/%d)…", sym, i + 1, n)
                try:
                    df = self.get_historical_data(
                        sym, exchange=exchange, currency=currency, duration=duration
                    )
                    if not df.empty and "close" in df.columns:
                        close_series[sym] = df["close"]
                except Exception as exc:
                    logger.warning("Skipping %s: %s", sym, exc)
            if not close_series:
                return pd.DataFrame()
            return pd.DataFrame(close_series).sort_index()

        # ── Account / positions ───────────────────────────────────────────

        def get_nav(self, timeout: float = 10.0) -> float:
            """Return total account Net Liquidation Value in base currency."""
            self._wrapper._acct_done.clear()
            self._wrapper.account_summary.clear()
            self._client.reqAccountSummary(9001, "All", "NetLiquidation")
            self._wrapper._acct_done.wait(timeout=timeout)
            val, _ = self._wrapper.account_summary.get("NetLiquidation", (None, None))
            if val is None:
                raise RuntimeError("Could not retrieve NetLiquidation from IBKR")
            return float(val)

        def get_positions(self, timeout: float = 10.0) -> dict:
            """Return current stock positions as {symbol: shares (int)}."""
            self._wrapper._positions.clear()
            self._wrapper._pos_done.clear()
            self._client.reqPositions()
            self._wrapper._pos_done.wait(timeout=timeout)
            self._client.cancelPositions()
            return dict(self._wrapper._positions)

        # ── Order placement ───────────────────────────────────────────────

        def _next_order_id(self) -> int:
            oid = self._wrapper.next_order_id + self._order_counter
            self._order_counter += 1
            return oid

        def place_order(
            self,
            symbol: str,
            action: str,              # "BUY" or "SELL"
            quantity: int,
            order_type: str = "MOC",  # MOC = Market On Close (best for monthly)
            lmt_price: float = None,
            exchange: str = "SMART",
            currency: str = "USD",
            dry_run: bool = True,
        ) -> Optional[int]:
            """
            Place a BUY or SELL order.

            dry_run=True (default): log the order but do NOT send it.
            order_type='MOC': Market On Close — send before 3:45 PM ET on the
                last trading day of the month; fills at the closing price.
            order_type='MKT': immediate market order (use during market hours only).
            """
            if quantity <= 0:
                return None
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = exchange
            contract.currency = currency

            order = Order()
            order.action = action.upper()
            order.orderType = order_type
            order.totalQuantity = quantity
            if order_type == "LMT" and lmt_price is not None:
                order.lmtPrice = round(lmt_price, 2)
            order.transmit = True

            order_id = self._next_order_id()
            tag = "[DRY-RUN] " if dry_run else ""
            logger.info("%s%s %d %s @ %s (orderId=%s)",
                        tag, action.upper(), quantity, symbol, order_type, order_id)

            if not dry_run:
                self._client.placeOrder(order_id, contract, order)

            return order_id

else:
    class IBKRConnection:  # type: ignore[no-redef]
        """Stub when ibapi is not installed."""
        def __init__(self, **kwargs):
            raise RuntimeError("ibapi is not installed. Run: pip install ibapi")
