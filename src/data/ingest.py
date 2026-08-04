"""
Downloads REAL historical OHLCV data from Yahoo Finance via `yfinance`.
No API key required. Can be run standalone or called as a ZenML step.
"""
import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf

from src.utils.config import load_config, resolve_path


def download_ticker(ticker: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """Download raw OHLCV data for a single ticker."""
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol / dates / network.")
    # yfinance sometimes returns MultiIndex columns for a single ticker depending on version
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Ticker"] = ticker
    df.columns = [str(c).strip() for c in df.columns]
    return df


def download_all(tickers: List[str], start_date: str, end_date: Optional[str] = None,
                  save: bool = True) -> pd.DataFrame:
    """Download data for a list of tickers and concatenate into one dataframe."""
    frames = []
    for t in tickers:
        print(f"[ingest] downloading real market data for {t} ...")
        frames.append(download_ticker(t, start_date, end_date))
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if save:
        cfg = load_config()
        out_dir = resolve_path(cfg["paths"]["raw_data_dir"])
        out_path = os.path.join(out_dir, "market_data.csv")
        full.to_csv(out_path, index=False)
        print(f"[ingest] saved {len(full)} rows -> {out_path}")

    return full


def load_raw() -> pd.DataFrame:
    """Load previously downloaded raw data from disk."""
    cfg = load_config()
    out_dir = resolve_path(cfg["paths"]["raw_data_dir"])
    path = os.path.join(out_dir, "market_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No raw data found at {path}. Run `python -m src.data.ingest` first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


if __name__ == "__main__":
    cfg = load_config()
    download_all(cfg["tickers"], cfg["start_date"], cfg.get("end_date"))
