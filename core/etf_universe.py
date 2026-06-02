"""
Survivorship-bias-free ETF universe for Ireland-based IBKR accounts.

WHY ETFs INSTEAD OF INDIVIDUAL STOCKS?
  - No survivorship bias: an ETF tracks an index; it can't "go bankrupt"
  - No look-ahead bias in universe construction
  - All are directly tradeable on Euronext Dublin or LSE via IBKR Ireland
  - USD-denominated ETFs available too (but EUR-hedged versions preferred in IE)

HOW TO USE
----------
  1. Download via yfinance on your local machine (not available in sandbox):
       from core.data_fetcher import get_data
       df = get_data("CSPX.L", source="yfinance", period="5y")

  2. Run any strategy using these as your 'tickers' list (no survivorship bias!)
"""

# ── European/global ETFs listed on LSE (GBP-denominated) ──────────────────
# All tradeable on Euronext or LSE via IBKR Ireland
LSE_ETFS = {
    # Broad equity
    "CSPX.L":  "iShares Core S&P 500 (USD, LSE)",
    "IWQU.L":  "iShares MSCI World Quality Factor",
    "SWRD.L":  "SPDR MSCI World (USD, low cost)",
    "VWRL.L":  "Vanguard FTSE All-World",

    # Sector ETFs (for sector rotation strategies)
    "IUIT.L":  "iShares S&P 500 IT Sector",
    "INRG.L":  "iShares Global Clean Energy",
    "IGLN.L":  "iShares Physical Gold ETC",

    # Fixed income (for multi-asset strategies)
    "IGIL.L":  "iShares Global Inflation-Linked Govt Bond",
    "IDTM.L":  "iShares $ Treasury Bond 7-10yr",

    # Factor ETFs
    "IWMO.L":  "iShares MSCI World Momentum Factor",
    "IWVL.L":  "iShares MSCI World Value Factor",
    "MVOL.L":  "iShares MSCI World Min Volatility",
}

# ── Euronext Dublin (EUR-denominated, good for IE investors) ──────────────
EURONEXT_DUBLIN_ETFS = {
    "CSPX.DE":  "iShares Core S&P 500 (EUR-hedged equivalent on Xetra)",
    "EXS1.DE":  "iShares Core DAX",
    "EXSA.DE":  "iShares Core EURO STOXX 50",
}

# ── US-listed (available via IBKR, USD) ─────────────────────────────────
US_ETFS = {
    "SPY":  "SPDR S&P 500",
    "QQQ":  "Invesco Nasdaq-100",
    "IWM":  "iShares Russell 2000 (small-cap)",
    "EFA":  "iShares MSCI EAFE (developed ex-US)",
    "EEM":  "iShares MSCI Emerging Markets",
    "TLT":  "iShares 20+ Year Treasury Bond",
    "GLD":  "SPDR Gold Shares",
    # Sector SPDRs (good for cross-sectional sector rotation)
    "XLK":  "Technology Select Sector SPDR",
    "XLF":  "Financial Select Sector SPDR",
    "XLE":  "Energy Select Sector SPDR",
    "XLV":  "Health Care Select Sector SPDR",
    "XLI":  "Industrial Select Sector SPDR",
    "XLP":  "Consumer Staples Select Sector SPDR",
    "XLY":  "Consumer Discretionary Select Sector SPDR",
    "XLRE": "Real Estate Select Sector SPDR",
}

# ── Strategy-focused collections ─────────────────────────────────────────

def sector_rotation_universe():
    """11 US sector ETFs — classic cross-sectional momentum / sector rotation."""
    return ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLRE",
            "XLU", "XLB", "XLC"]

def multi_asset_universe():
    """Broad multi-asset (equity + bond + commodity) — 'risk parity' style."""
    return ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"]

def factor_etf_universe():
    """Factor ETFs from iShares — test factor premiums without single-stock noise."""
    return ["IWMO.L", "IWVL.L", "MVOL.L", "IWQU.L", "VWRL.L"]

def ireland_liquid_universe():
    """Liquid ETFs accessible from IBKR Ireland with reasonable AUM."""
    return list(LSE_ETFS.keys())


# ── IBKR contract specs for the most common ones ─────────────────────────
# Pass these kwargs to IBKRConnection.get_historical_data()
IBKR_SPECS = {
    "CSPX.L": {"exchange": "LSE",   "currency": "USD", "sec_type": "STK"},
    "VWRL.L": {"exchange": "LSE",   "currency": "GBP", "sec_type": "STK"},
    "SPY":    {"exchange": "SMART", "currency": "USD", "sec_type": "STK"},
    "QQQ":    {"exchange": "SMART", "currency": "USD", "sec_type": "STK"},
}
