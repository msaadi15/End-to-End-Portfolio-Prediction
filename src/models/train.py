"""
Candidate model definitions and evaluation helpers.

Evaluation combines standard regression metrics (RMSE, MAE, R2) with
finance-specific metrics that actually matter for a portfolio:
  - directional accuracy: did we correctly predict up vs down?
  - backtested Sharpe ratio: if we go long when predicted return > 0 and
    flat otherwise, what's the risk-adjusted return of that simple strategy?
  - cumulative return of that simple strategy over the test period.
"""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def get_candidate_models(random_state: int = 42) -> Dict[str, object]:
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=random_state,
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state,
            objective="reg:squarederror",
        )
    return models


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def backtest_long_flat(y_true: np.ndarray, y_pred: np.ndarray,
                        risk_free_rate: float = 0.02,
                        periods_per_year: int = 252) -> Dict[str, float]:
    """
    Very simple long/flat strategy: go long (position=1) when the model predicts
    a positive next-day return, stay flat (position=0) otherwise.
    Returns realized strategy returns' Sharpe ratio and cumulative return.
    """
    position = (y_pred > 0).astype(float)
    strategy_returns = position * y_true

    if strategy_returns.std() == 0 or len(strategy_returns) < 2:
        sharpe = 0.0
    else:
        daily_rf = risk_free_rate / periods_per_year
        excess = strategy_returns - daily_rf
        sharpe = float((excess.mean() / (excess.std() + 1e-12)) * np.sqrt(periods_per_year))

    cum_return = float(np.prod(1 + strategy_returns) - 1)
    return {"sharpe": sharpe, "cum_return": cum_return}


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_true = y_test.values

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    dir_acc = directional_accuracy(y_true, y_pred)
    bt = backtest_long_flat(y_true, y_pred, risk_free_rate=risk_free_rate)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "dir_acc": dir_acc,
        "sharpe": bt["sharpe"],
        "cum_return": bt["cum_return"],
        "n_test_samples": int(len(y_true)),
    }


def compute_champion_score(metrics: Dict[str, float], all_metrics: Dict[str, Dict[str, float]],
                            w_rmse: float = 0.3, w_sharpe: float = 0.4, w_diracc: float = 0.3) -> float:
    """
    Blend metrics into a single comparable score (higher is better).
    RMSE is inverted+normalized across all candidates so lower RMSE -> higher score.
    """
    rmses = [m["rmse"] for m in all_metrics.values()]
    sharpes = [m["sharpe"] for m in all_metrics.values()]

    rmse_min, rmse_max = min(rmses), max(rmses)
    sharpe_min, sharpe_max = min(sharpes), max(sharpes)

    def norm(val, lo, hi, invert=False):
        if hi - lo < 1e-12:
            return 0.5
        n = (val - lo) / (hi - lo)
        return 1 - n if invert else n

    rmse_score = norm(metrics["rmse"], rmse_min, rmse_max, invert=True)
    sharpe_score = norm(metrics["sharpe"], sharpe_min, sharpe_max, invert=False)
    diracc_score = metrics["dir_acc"]  # already 0-1

    return float(w_rmse * rmse_score + w_sharpe * sharpe_score + w_diracc * diracc_score)
