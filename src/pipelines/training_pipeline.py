"""
The full ZenML training pipeline. Run it directly:
    python -m src.pipelines.training_pipeline
or via scripts/run_pipeline.sh (recommended, also handles zenml init/up).

Open the ZenML dashboard (started by `zenml up`) to watch each step run live
and inspect artifacts/metrics: http://127.0.0.1:8237
"""
from zenml import pipeline

from src.pipelines.steps import (
    ingest_data_step,
    build_features_step,
    split_data_step,
    train_and_evaluate_step,
)


@pipeline(enable_cache=False)
def finance_training_pipeline():
    raw_df = ingest_data_step()
    feats = build_features_step(raw_df)
    X_train, X_test, y_train, y_test = split_data_step(feats)
    train_and_evaluate_step(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    finance_training_pipeline()
