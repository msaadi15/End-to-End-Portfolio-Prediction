"""
Live performance watchdog.

Periodically re-evaluates the CURRENT CHAMPION on the freshest data (re-downloads
recent data, rebuilds features, backtests the champion). If its rolling Sharpe
ratio drops below `sharpe_drop_threshold` (config.yaml), it will:

  1. Look at every other model already sitting in the registry and, if one of
     them currently performs better, promote it as the new champion instantly
     (no retraining needed - fast failover).
  2. If none of them beat the threshold either, trigger a full pipeline retrain
     on fresh data.
  3. Call POST /reload on the API so it hot-swaps to the new champion with zero
     downtime.

Usage:
    python src/monitoring/monitor.py            # run forever, checking on schedule
    python src/monitoring/monitor.py --once      # run a single check and exit
"""
import argparse
import time
from datetime import datetime

import requests

from src.data.ingest import download_all
from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN, build_features
from src.models.train import backtest_long_flat
from src.utils.config import load_config
from src.utils.registry import get_champion_info, list_models, load_metrics, load_model, set_champion


def _log(msg: str):
    print(f"[monitor {datetime.utcnow().isoformat()}Z] {msg}")


def check_and_switch_if_needed(cfg: dict):
    champ_info = get_champion_info()
    champion_name = champ_info.get("champion")
    if not champion_name:
        _log("No champion registered yet — skipping check. Run the training pipeline first.")
        return

    _log(f"Re-downloading fresh data to evaluate live performance of champion '{champion_name}'...")
    raw = download_all(cfg["tickers"], cfg["start_date"], cfg.get("end_date"), save=False)
    feats = build_features(raw)
    window = cfg.get("sharpe_window", 60)
    recent = feats.sort_values("Date").groupby("Ticker").tail(window)

    X_recent = recent[FEATURE_COLUMNS]
    y_recent = recent[TARGET_COLUMN].values

    champion_model = load_model(champion_name)
    y_pred = champion_model.predict(X_recent)
    bt = backtest_long_flat(y_recent, y_pred, risk_free_rate=cfg.get("risk_free_rate", 0.02))
    live_sharpe = bt["sharpe"]
    threshold = cfg.get("sharpe_drop_threshold", 0.3)

    _log(f"Champion '{champion_name}' live rolling Sharpe = {live_sharpe:.3f} (threshold={threshold})")

    if live_sharpe >= threshold:
        _log("Performance is healthy. No action needed.")
        return

    _log(f"⚠️ Performance DROPPED below threshold! Searching for a better challenger...")

    best_alt_name, best_alt_sharpe = None, live_sharpe
    for name in list_models():
        if name == champion_name:
            continue
        try:
            alt_model = load_model(name)
            alt_pred = alt_model.predict(X_recent)
            alt_bt = backtest_long_flat(y_recent, alt_pred, risk_free_rate=cfg.get("risk_free_rate", 0.02))
            _log(f"  Candidate '{name}' live Sharpe = {alt_bt['sharpe']:.3f}")
            if alt_bt["sharpe"] > best_alt_sharpe:
                best_alt_name, best_alt_sharpe = name, alt_bt["sharpe"]
        except FileNotFoundError:
            continue

    if best_alt_name and best_alt_sharpe >= threshold:
        set_champion(
            best_alt_name,
            reason=f"auto-switch: champion '{champion_name}' Sharpe fell to {live_sharpe:.3f}, "
                   f"'{best_alt_name}' scored {best_alt_sharpe:.3f} on live data",
        )
        _log(f"✅ Promoted '{best_alt_name}' as new champion (live Sharpe {best_alt_sharpe:.3f}).")
        _notify_api_reload(cfg)
        return

    _log("No existing candidate beats the threshold either. Triggering full retrain on fresh data...")
    from src.pipelines.training_pipeline import finance_training_pipeline
    finance_training_pipeline()
    _log("Retrain complete — a new champion has been selected by the pipeline.")
    _notify_api_reload(cfg)


def _notify_api_reload(cfg: dict):
    url = cfg.get("api_base_url", "http://127.0.0.1:8000") + "/reload"
    try:
        r = requests.post(url, timeout=5)
        _log(f"Notified API to reload champion -> {r.status_code}: {r.json()}")
    except requests.exceptions.RequestException as e:
        _log(f"Could not reach API to reload (is it running?): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single check and exit")
    args = parser.parse_args()

    cfg = load_config()
    interval = cfg.get("check_interval_minutes", 60) * 60

    if args.once:
        check_and_switch_if_needed(cfg)
        return

    _log(f"Starting monitoring loop (checking every {interval // 60} minutes). Ctrl+C to stop.")
    while True:
        try:
            check_and_switch_if_needed(cfg)
        except Exception as e:
            _log(f"Error during check: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
