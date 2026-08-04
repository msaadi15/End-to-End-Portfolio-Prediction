"""
Builds quant/technical features from raw OHLCV data and the prediction target
(next-day return per ticker). All features are computed per-ticker to avoid
leaking information across different stocks.
"""
import os

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d",
    "volatility_10d", "volatility_20d",
    "sma_10", "sma_20", "sma_ratio",
    "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal",
    "momentum_10", "volume_change",
]
TARGET_COLUMN = "target_next_return"


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _build_for_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("Date").copy()
    close = g["Close"]

    g["return_1d"] = close.pct_change(1)
    g["return_5d"] = close.pct_change(5)
    g["return_10d"] = close.pct_change(10)

    g["volatility_10d"] = g["return_1d"].rolling(10).std()
    g["volatility_20d"] = g["return_1d"].rolling(20).std()

    g["sma_10"] = close.rolling(10).mean()
    g["sma_20"] = close.rolling(20).mean()
    g["sma_ratio"] = g["sma_10"] / g["sma_20"]

    g["ema_12"] = close.ewm(span=12, adjust=False).mean()
    g["ema_26"] = close.ewm(span=26, adjust=False).mean()
    g["macd"] = g["ema_12"] - g["ema_26"]
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()

    g["rsi_14"] = _rsi(close, 14)
    g["momentum_10"] = close - close.shift(10)

    if "Volume" in g.columns:
        g["volume_change"] = g["Volume"].pct_change(1)
    else:
        g["volume_change"] = 0.0

    # Target: next day's return (what we want to predict)
    g[TARGET_COLUMN] = close.pct_change(1).shift(-1)

    return g


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    processed = raw_df.groupby("Ticker", group_keys=False).apply(_build_for_ticker)
    processed = processed.replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    return processed


def save_processed(df: pd.DataFrame) -> str:
    cfg = load_config()
    out_dir = resolve_path(cfg["paths"]["processed_data_dir"])
    path = os.path.join(out_dir, "features.csv")
    df.to_csv(path, index=False)
    print(f"[features] saved {len(df)} rows -> {path}")
    return path


def load_processed() -> pd.DataFrame:
    cfg = load_config()
    out_dir = resolve_path(cfg["paths"]["processed_data_dir"])
    path = os.path.join(out_dir, "features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No processed features found at {path}. Run the pipeline first.")
    return pd.read_csv(path, parse_dates=["Date"])


if __name__ == "__main__":
    from src.data.ingest import load_raw
    raw = load_raw()
    feats = build_features(raw)
    save_processed(feats)
