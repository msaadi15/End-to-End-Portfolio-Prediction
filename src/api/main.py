"""
FastAPI serving layer for the current champion model.

Endpoints:
  GET  /health    -> liveness check
  GET  /models     -> list all trained candidate models + their metrics
  GET  /metrics    -> metrics of the current champion
  POST /predict    -> predict next-day return for a ticker using latest features
  POST /reload     -> hot-reload champion from disk (called by the monitor after a switch)
"""
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features.build_features import FEATURE_COLUMNS, load_processed
from src.utils.registry import get_champion_info, list_models, load_metrics, load_model

app = FastAPI(
    title="Finance MLOps Portfolio API",
    description="Serves predictions from the current champion model, with live monitoring & auto model-switching.",
    version="1.0.0",
)

_state = {"model": None, "champion_name": None}


def _load_champion():
    info = get_champion_info()
    name = info.get("champion")
    if not name:
        raise RuntimeError("No champion registered yet. Run the training pipeline first.")
    _state["model"] = load_model(name)
    _state["champion_name"] = name
    return name


@app.on_event("startup")
def startup_event():
    try:
        name = _load_champion()
        print(f"[api] loaded champion model: {name}")
    except Exception as e:
        print(f"[api] WARNING: could not load champion at startup: {e}")


class PredictRequest(BaseModel):
    ticker: str
    features: Optional[dict] = None  # optional manual feature override


@app.get("/health")
def health():
    return {
        "status": "ok" if _state["model"] is not None else "no_model_loaded",
        "champion": _state["champion_name"],
    }


@app.get("/models")
def models():
    return {"available_models": list_models(), "metrics": load_metrics(), "champion": get_champion_info()}


@app.get("/metrics")
def metrics():
    all_metrics = load_metrics()
    champ = _state["champion_name"]
    if champ and champ in all_metrics:
        return {"champion": champ, "metrics": all_metrics[champ]}
    return {"champion": champ, "metrics": None}


@app.post("/reload")
def reload_champion():
    try:
        name = _load_champion()
        return {"reloaded": True, "champion": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded yet. Train the pipeline first.")

    if req.features:
        row = pd.DataFrame([req.features])[FEATURE_COLUMNS]
    else:
        try:
            feats = load_processed()
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="No processed data found. Run the pipeline first.")
        ticker_rows = feats[feats["Ticker"] == req.ticker.upper()]
        if ticker_rows.empty:
            raise HTTPException(status_code=404, detail=f"No data for ticker '{req.ticker}'.")
        row = ticker_rows.sort_values("Date").iloc[[-1]][FEATURE_COLUMNS]

    pred = float(_state["model"].predict(row)[0])
    return {
        "ticker": req.ticker.upper(),
        "predicted_next_day_return": pred,
        "signal": "LONG" if pred > 0 else "FLAT",
        "champion_model": _state["champion_name"],
    }
