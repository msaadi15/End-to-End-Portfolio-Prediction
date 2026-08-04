"""
Basic smoke tests. Run with: pytest tests/
These don't require network access except test_ingest_shapes (marked accordingly).
"""
import numpy as np
import pandas as pd

from src.models.train import backtest_long_flat, directional_accuracy, compute_champion_score


def test_directional_accuracy_perfect():
    y_true = np.array([0.01, -0.02, 0.03, -0.01])
    y_pred = np.array([0.02, -0.01, 0.01, -0.02])
    assert directional_accuracy(y_true, y_pred) == 1.0


def test_directional_accuracy_worst():
    y_true = np.array([0.01, -0.02, 0.03, -0.01])
    y_pred = -y_true
    assert directional_accuracy(y_true, y_pred) == 0.0


def test_backtest_long_flat_shapes():
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 0.01, 100)
    y_pred = rng.normal(0, 0.01, 100)
    result = backtest_long_flat(y_true, y_pred)
    assert "sharpe" in result and "cum_return" in result
    assert isinstance(result["sharpe"], float)


def test_compute_champion_score_picks_best():
    all_metrics = {
        "A": {"rmse": 0.01, "sharpe": 1.0, "dir_acc": 0.6},
        "B": {"rmse": 0.02, "sharpe": 0.2, "dir_acc": 0.5},
    }
    score_a = compute_champion_score(all_metrics["A"], all_metrics)
    score_b = compute_champion_score(all_metrics["B"], all_metrics)
    assert score_a > score_b


def test_feature_columns_consistency():
    from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
