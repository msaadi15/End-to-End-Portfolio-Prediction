"""
Lightweight model registry (no external DB needed).

Layout inside models_registry/:
    <ModelName>.pkl            trained sklearn/xgboost model, one per candidate
    metrics.json                {model_name: {rmse, mae, r2, dir_acc, sharpe, cum_return, score}}
    champion.json                {"champion": "<ModelName>", "promoted_at": iso_ts, "reason": "..."}
"""
import json
import os
import shutil
import time
from datetime import datetime

import joblib

from src.utils.config import load_config, resolve_path


def registry_dir() -> str:
    cfg = load_config()
    d = resolve_path(cfg["paths"]["registry_dir"])
    os.makedirs(d, exist_ok=True)
    return d


def save_model(model, name: str):
    path = os.path.join(registry_dir(), f"{name}.pkl")
    joblib.dump(model, path)
    return path


def load_model(name: str):
    path = os.path.join(registry_dir(), f"{name}.pkl")
    return joblib.load(path)


def save_metrics(all_metrics: dict):
    path = os.path.join(registry_dir(), "metrics.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)


def load_metrics() -> dict:
    path = os.path.join(registry_dir(), "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def set_champion(name: str, reason: str = "best score on training-time evaluation"):
    path = os.path.join(registry_dir(), "champion.json")
    payload = {
        "champion": name,
        "promoted_at": datetime.utcnow().isoformat() + "Z",
        "reason": reason,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    # keep a convenience copy so the API can always load "champion.pkl" directly
    champ_src = os.path.join(registry_dir(), f"{name}.pkl")
    champ_dst = os.path.join(registry_dir(), "champion.pkl")
    if os.path.exists(champ_src):
        shutil.copyfile(champ_src, champ_dst)
    return payload


def get_champion_info() -> dict:
    path = os.path.join(registry_dir(), "champion.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def list_models() -> list:
    d = registry_dir()
    if not os.path.exists(d):
        return []
    return sorted(
        f[:-4] for f in os.listdir(d)
        if f.endswith(".pkl") and f != "champion.pkl"
    )
