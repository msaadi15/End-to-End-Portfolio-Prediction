"""
ZenML steps. Each @step shows up as a node in the ZenML dashboard, so you can
watch data flow through the pipeline live and inspect logs/metrics per step.
"""
from typing import Tuple

import pandas as pd
from zenml import step
from zenml.logger import get_logger

from src.data.ingest import download_all
from src.features.build_features import build_features, save_processed, FEATURE_COLUMNS, TARGET_COLUMN
from src.models.train import get_candidate_models, evaluate, compute_champion_score
from src.utils.config import load_config
from src.utils.registry import save_model, save_metrics, set_champion

logger = get_logger(__name__)


@step
def ingest_data_step() -> pd.DataFrame:
    """Downloads real OHLCV data for all configured tickers."""
    cfg = load_config()
    df = download_all(cfg["tickers"], cfg["start_date"], cfg.get("end_date"))
    logger.info(f"Downloaded {len(df)} raw rows across {df['Ticker'].nunique()} tickers.")
    return df


@step
def build_features_step(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Builds technical/quant features + next-day-return target."""
    feats = build_features(raw_df)
    save_processed(feats)
    logger.info(f"Built {len(feats)} feature rows with {len(FEATURE_COLUMNS)} features.")
    return feats


@step
def split_data_step(feats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-based split (no shuffling!) to avoid lookahead bias in finance data."""
    cfg = load_config()
    test_size = cfg.get("test_size", 0.2)

    feats = feats.sort_values("Date")
    split_idx = int(len(feats) * (1 - test_size))

    train_df = feats.iloc[:split_idx]
    test_df = feats.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    logger.info(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    return X_train, X_test, y_train, y_test


@step
def train_and_evaluate_step(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> str:
    """
    Trains every candidate model, evaluates it with finance-aware metrics,
    saves each one to the registry, computes a blended champion score, and
    promotes the best model as the new champion. Returns champion name.
    """
    cfg = load_config()
    models = get_candidate_models()

    all_metrics = {}
    fitted_models = {}

    for name, model in models.items():
        logger.info(f"Training {name} ...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, risk_free_rate=cfg.get("risk_free_rate", 0.02))
        all_metrics[name] = metrics
        fitted_models[name] = model
        save_model(model, name)
        logger.info(
            f"{name} -> RMSE={metrics['rmse']:.5f} DirAcc={metrics['dir_acc']:.3f} "
            f"Sharpe={metrics['sharpe']:.3f} CumReturn={metrics['cum_return']:.3f}"
        )

    for name, metrics in all_metrics.items():
        metrics["score"] = compute_champion_score(
            metrics, all_metrics,
            w_rmse=cfg.get("weight_rmse", 0.3),
            w_sharpe=cfg.get("weight_sharpe", 0.4),
            w_diracc=cfg.get("weight_diracc", 0.3),
        )

    save_metrics(all_metrics)

    champion_name = max(all_metrics, key=lambda n: all_metrics[n]["score"])
    set_champion(champion_name, reason="highest blended score (RMSE+Sharpe+DirAcc) on holdout set")

    logger.info(f"🏆 New champion: {champion_name} (score={all_metrics[champion_name]['score']:.4f})")
    _print_leaderboard(all_metrics, champion_name)

    return champion_name


def _print_leaderboard(all_metrics: dict, champion_name: str):
    try:
        from tabulate import tabulate
        rows = []
        for name, m in sorted(all_metrics.items(), key=lambda kv: -kv[1]["score"]):
            tag = " <- CHAMPION" if name == champion_name else ""
            rows.append([name + tag, f"{m['rmse']:.5f}", f"{m['dir_acc']:.3f}",
                         f"{m['sharpe']:.3f}", f"{m['cum_return']:.3f}", f"{m['score']:.4f}"])
        print("\n" + tabulate(rows, headers=["Model", "RMSE", "DirAcc", "Sharpe", "CumReturn", "Score"]))
    except ImportError:
        print(all_metrics)
