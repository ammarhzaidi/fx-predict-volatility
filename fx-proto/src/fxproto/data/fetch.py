# src/fxproto/data/fetch.py
from __future__ import annotations
import os
from pathlib import Path
from datetime import date
from typing import Optional, Literal

import pandas as pd
import yfinance as yf

# --- Defaults (can later be read from config) ---
DEFAULT_DATA_DIR = Path("fx-proto") / "data" / "raw"
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Yahoo symbols for common FX pairs
YF_SYMBOL_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCAD": "CAD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCHF": "CHF=X",
}

def _resolve_symbol(symbol: str) -> str:
    """Accepts 'EURUSD' or already 'EURUSD=X'; returns a Yahoo-ready symbol."""
    symbol = symbol.strip().upper()
    if symbol in YF_SYMBOL_MAP:
        return YF_SYMBOL_MAP[symbol]
    # If user already passed a Yahoo-style symbol, keep it
    if symbol.endswith("=X"):
        return symbol
    raise ValueError(f"Unrecognized symbol '{symbol}'. "
                     f"Use one of {list(YF_SYMBOL_MAP)} or a Yahoo ticker like 'EURUSD=X'.")

def fetch_ohlcv(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: Literal["1d", "1h", "30m", "15m", "5m", "1m"] = "1d",
    save_csv: bool = True,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Yahoo Finance and return a tidy DataFrame indexed by datetime.

    Parameters
    ----------
    symbol : str
        FX pair, e.g. 'EURUSD' or Yahoo 'EURUSD=X'.
    start, end : 'YYYY-MM-DD' or None
        Date range (inclusive start, exclusive-ish end per yfinance). If None, yfinance picks defaults.
    interval : str
        Sampling interval (default '1d').
    save_csv : bool
        If True, write a CSV snapshot under data/raw/.
    data_dir : Path
        Where to save CSVs.

    Returns
    -------
    pd.DataFrame with columns: [Open, High, Low, Close, Adj Close?, Volume]
    """
    yf_symbol = _resolve_symbol(symbol)

    df = yf.download(yf_symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {yf_symbol} (range={start}..{end}, interval={interval}).")

    # Standardize index/columns
    df.index.name = "Date"
    # Some FX tickers have zero Volume; keep column for consistency
    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[cols].copy()

    # Basic cleaning: drop duplicates, sort, drop NA rows
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(how="any")

    if save_csv:
        # Example: EURUSD_1d_2024-06-01_2024-09-01.csv
        start_tag = start or "min"
        end_tag = end or "max"
        fname = f"{symbol.upper()}_{interval}_{start_tag}_{end_tag}.csv".replace("=", "")
        out_path = Path(data_dir) / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=True)
        # tiny breadcrumb for downstream logs
        print(f"[fetch_ohlcv] Saved {len(df):,} rows to {out_path}")

    return df
