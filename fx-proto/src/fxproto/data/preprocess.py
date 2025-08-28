from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fxproto.config.loader import get_config, resolved_data_dir

def load_or_fetch(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Load from csv in data/raw if available; else fetch via yfinance and save."""
    from fxproto.data.fetch import fetch_ohlcv  # local import to avoid cycles
    raw_dir = resolved_data_dir() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{symbol}_{interval}_{start or 'min'}_{end or 'max'}.csv".replace("=", "")
    fpath = raw_dir / fname
    if fpath.exists():
        df = pd.read_csv(fpath, parse_dates=["Date"], index_col="Date")
    else:
        df = fetch_ohlcv(symbol, start=start, end=end, interval=interval, save_csv=True)
    return df

def train_test_split_by_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = get_config().settings.dates
    train = df.loc[cfg.train_start:cfg.train_end].copy()
    test = df.loc[cfg.test_start:cfg.test_end].copy()
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Empty train/test slice — check dates in config/settings.yaml")
    return train, test

def scale_train_apply_test(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    scaler = StandardScaler()
    train[cols] = scaler.fit_transform(train[cols].values)
    test[cols]  = scaler.transform(test[cols].values)
    return train, test, scaler
