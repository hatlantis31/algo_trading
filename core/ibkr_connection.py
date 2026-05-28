"""
Thin wrapper around ibapi to manage TWS / IB Gateway connections.

Usage:
    conn = IBKRConnection(port=7497)   # paper TWS
    conn.connect()
    df = conn.get_historical_data("AAPL", exchange="SMART", currency="USD")
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
    from ibapi.common import BarData
    IBAPI_AVAILABLE = True
except ImportError:
    IBAPI_AVAILABLE = False
    logging.warning("ibapi not installed – IBKR live connection unavailable.")

from config.settings import IBKR_HOST, IBKR_TWS_PORT, IBKR_CLIENT_ID

logger = logging.getLogger(__name__)


if IBAPI_AVAILABLE:
    class _IBWrapper(EWrapper):
        def __init__(self):
            super().__init__()
            self._bars: list[BarData] = []
            self._done = threading.Event()
            self.next_order_id: Optional[int] = None
            self.account_summary: dict = {}

        def historicalData(self, reqId, bar):
            self._bars.append(bar)

        def historicalDataEnd(self, reqId, start, end):
            self._done.set()

        def nextValidId(self, orderId):
            self.next_order_id = orderId

        def accountSummary(self, reqId, account, tag, value, currency):
            self.account_summary[tag] = (value, currency)

        def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
            if errorCode not in (2104, 2106, 2158):  # benign info codes
                logger.error("IBKR error %s (reqId %s): %s", errorCode, reqId, errorString)

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

        def connect(self, timeout: float = 10.0) -> bool:
            self._client.connect(self.host, self.port, self.client_id)
            self._thread = threading.Thread(target=self._client.run, daemon=True)
            self._thread.start()
            deadline = time.time() + timeout
            while self._wrapper.next_order_id is None and time.time() < deadline:
                time.sleep(0.1)
            connected = self._wrapper.next_order_id is not None
            if connected:
                logger.info("Connected to IBKR at %s:%s", self.host, self.port)
            else:
                logger.error("Failed to connect within %.1fs", timeout)
            return connected

        def disconnect(self):
            self._client.disconnect()
            logger.info("Disconnected from IBKR")

        def get_historical_data(
            self,
            symbol: str,
            exchange: str = "SMART",
            currency: str = "EUR",
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
            self._wrapper._done.clear()
            req_id = 1
            end_dt = datetime.now().strftime("%Y%m%d %H:%M:%S")
            self._client.reqHistoricalData(
                req_id, contract, end_dt, duration, bar_size,
                what_to_show, 1, 1, False, []
            )
            self._wrapper._done.wait(timeout=30)

            rows = [
                {
                    "date": b.date,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in self._wrapper._bars
            ]
            df = pd.DataFrame(rows)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            return df

else:
    class IBKRConnection:  # type: ignore[no-redef]
        """Stub when ibapi is not installed."""
        def __init__(self, **kwargs):
            raise RuntimeError("ibapi is not installed. Run: pip install ibapi")
