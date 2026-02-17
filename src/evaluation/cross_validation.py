"""Cross-validation evaluation with overfit detection."""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

_DEFAULT_SCORING = {
    "accuracy": "accuracy",
    "f1_weighted": "f1_weighted",
    "f1_macro": "f1_macro",
}


def evaluate_with_cv(
    pipeline: Pipeline,
    X,
    y,
    cv_folds: int = 5,
    scoring: dict | None = None,
) -> dict:
    """Evaluate a pipeline with cross-validation and detect overfitting.

    Returns
    -------
    dict with keys:
        per_fold : pd.DataFrame  -- per-fold train/test scores
        summary  : dict          -- mean and std of each metric
        overfit_flags : dict      -- True if train-test gap > 2 * test_std
    """
    if scoring is None:
        scoring = _DEFAULT_SCORING

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=True,
    )

    # Build per-fold DataFrame
    fold_data = {}
    for metric_name in scoring:
        train_key = f"train_{metric_name}"
        test_key = f"test_{metric_name}"
        fold_data[f"train_{metric_name}"] = cv_results[train_key]
        fold_data[f"test_{metric_name}"] = cv_results[test_key]
    per_fold = pd.DataFrame(fold_data)

    # Summary statistics
    summary = {}
    overfit_flags = {}
    for metric_name in scoring:
        train_scores = cv_results[f"train_{metric_name}"]
        test_scores = cv_results[f"test_{metric_name}"]
        train_mean = float(np.mean(train_scores))
        test_mean = float(np.mean(test_scores))
        test_std = float(np.std(test_scores))

        summary[metric_name] = {
            "train_mean": train_mean,
            "test_mean": test_mean,
            "test_std": test_std,
            "gap": train_mean - test_mean,
        }
        # Overfit flag: gap > 2 * test_std (with floor to avoid false
        # positives when test_std is near zero)
        effective_std = max(test_std, 0.01)
        overfit_flags[metric_name] = (train_mean - test_mean) > 2 * effective_std

    return {
        "per_fold": per_fold,
        "summary": summary,
        "overfit_flags": overfit_flags,
    }


def compare_models_cv(
    model_configs: dict,
    X,
    y,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Compare multiple model pipelines using cross-validation.

    Parameters
    ----------
    model_configs : dict[str, Pipeline]
        Mapping of model name to (unfitted) Pipeline.

    Returns
    -------
    pd.DataFrame with summary statistics per model.
    """
    rows = []
    for name, pipe in model_configs.items():
        result = evaluate_with_cv(pipe, X, y, cv_folds=cv_folds)
        row = {"model": name}
        for metric_name, stats in result["summary"].items():
            row[f"{metric_name}_test_mean"] = stats["test_mean"]
            row[f"{metric_name}_test_std"] = stats["test_std"]
            row[f"{metric_name}_overfit"] = result["overfit_flags"][metric_name]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
