# fx-proto/src/fxproto/data/features.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced basic features including returns, moving averages, and volatility."""
    out = df.copy()

    # Price-based features
    out["ret_1"] = out["Close"].pct_change()
    out["ret_5"] = out["Close"].pct_change(5)
    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["vol_10"] = out["Close"].pct_change().rolling(10).std()
    out["vol_30"] = out["Close"].pct_change().rolling(30).std()

    # Technical indicators
    out["rsi_14"] = calculate_rsi(out["Close"], 14)
    out["bb_upper"], out["bb_lower"] = calculate_bollinger_bands(out["Close"])

    # Price momentum
    out["momentum_10"] = out["Close"] / out["Close"].shift(10) - 1
    out["price_range"] = (out["High"] - out["Low"]) / out["Close"]

    out = out.dropna()
    return out


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, std_dev: float = 2):
    """Calculate Bollinger Bands."""
    ma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    return upper, lower


def generate_synthetic_node_features(df: pd.DataFrame, pair: str = "EURUSD") -> pd.DataFrame:
    """
    Generate synthetic features for graph nodes based on price movements.
    This simulates data that would come from economic indicators, central bank sentiment, etc.
    """
    out = df.copy()

    # Generate synthetic node features based on price patterns
    price_change = out["Close"].pct_change()
    volatility = price_change.rolling(20).std()

    # ECB sentiment (correlated with EUR strength)
    if "EUR" in pair.upper():
        # ECB sentiment inversely correlated with EUR weakness
        ecb_base = -price_change * 2 + np.random.normal(0, 0.1, len(out))
        out["ecb_sentiment"] = pd.Series(ecb_base, index=out.index).rolling(5).mean()

    # Fed sentiment (correlated with USD strength)  
    if "USD" in pair.upper():
        # Fed sentiment positively correlated with USD strength (negative EUR/USD moves)
        fed_base = -price_change * 1.5 + np.random.normal(0, 0.1, len(out))
        out["fed_sentiment"] = pd.Series(fed_base, index=out.index).rolling(5).mean()

    # Interest rate differential (key FX driver)
    rate_trend = out["Close"].rolling(60).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    out["interest_rate_diff"] = rate_trend + np.random.normal(0, 0.05, len(out))

    # GDP growth differential 
    gdp_momentum = out["momentum_10"].rolling(20).mean()
    out["gdp_growth_diff"] = gdp_momentum + np.random.normal(0, 0.02, len(out))

    # Market risk sentiment (VIX-like)
    out["risk_sentiment"] = volatility * 100 + np.random.normal(0, 2, len(out))

    # Political stability index
    political_base = -abs(price_change) * 5 + np.random.normal(0.8, 0.1, len(out))
    out["political_stability"] = pd.Series(political_base, index=out.index).rolling(10).mean()

    return out.dropna()


def make_supervised_windows(df: pd.DataFrame, feature_cols: List[str], target_col: str,
                            lookback: int = 30, horizon: int = 5):
    """
    Create supervised learning windows for time series prediction.

    Returns:
        X: (N, lookback, F) - feature sequences
        y: (N,) - target values at t+horizon
    """
    # Ensure we have all required columns
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    values = df[feature_cols + [target_col]].values
    X, y = [], []

    for t in range(lookback, len(values) - horizon):
        # Features from t-lookback to t (exclusive)
        X.append(values[t - lookback:t, :len(feature_cols)])
        # Target at t+horizon
        y.append(values[t + horizon, -1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def get_feature_columns(include_synthetic: bool = True) -> Dict[str, List[str]]:
    """Return categorized feature column names."""
    price_features = ["ret_1", "ret_5", "ma_5", "ma_20", "vol_10", "vol_30"]
    technical_features = ["rsi_14", "bb_upper", "bb_lower", "momentum_10", "price_range"]

    feature_groups = {
        "price": price_features,
        "technical": technical_features,
    }

    if include_synthetic:
        synthetic_features = [
            "ecb_sentiment", "fed_sentiment", "interest_rate_diff",
            "gdp_growth_diff", "risk_sentiment", "political_stability"
        ]
        feature_groups["synthetic"] = synthetic_features

    return feature_groups


def prepare_model_data(df: pd.DataFrame, include_synthetic: bool = True) -> tuple[pd.DataFrame, List[str]]:
    """
    Complete data preparation pipeline returning enhanced dataframe and feature list.
    """
    # Apply basic features
    df_feat = basic_features(df)

    # Add synthetic node features
    if include_synthetic:
        df_feat = generate_synthetic_node_features(df_feat)

    # Get feature columns
    feature_groups = get_feature_columns(include_synthetic)
    all_features = []
    for group_features in feature_groups.values():
        all_features.extend(group_features)

    # Filter to only existing columns
    available_features = [col for col in all_features if col in df_feat.columns]

    return df_feat, available_features