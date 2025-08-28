from __future__ import annotations
import numpy as np
import pandas as pd

def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out["Close"].pct_change()
    out["ma_5"]  = out["Close"].rolling(5).mean()
    out["vol_10"] = out["Close"].pct_change().rolling(10).std()
    out = out.dropna()
    return out

def make_supervised_windows(df: pd.DataFrame, feature_cols: list[str], target_col: str,
                            lookback: int = 30, horizon: int = 5):
    """
    Return X (N, lookback, F), y (N,) predicting target_col[t + horizon].
    """
    values = df[feature_cols + [target_col]].values
    X, y = [], []
    for t in range(lookback, len(values)-horizon):
        X.append(values[t-lookback:t, :len(feature_cols)])
        y.append(values[t + horizon, -1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
