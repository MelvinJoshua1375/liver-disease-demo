"""Classification metrics, ROC analysis, and visualization helpers."""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute standard classification metrics.

    Returns dict with accuracy, f1_weighted, f1_macro, and optionally roc_auc.
    """
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    if y_proba is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_proba)
    return result


def compute_roc_data(y_true, y_proba) -> dict:
    """Compute ROC curve data including optimal threshold (Youden's J)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc,
        "optimal_threshold": float(thresholds[optimal_idx]),
    }


def compare_models(models_dict: dict, X_test, y_test) -> pd.DataFrame:
    """Compare multiple fitted pipelines on the test set.

    Parameters
    ----------
    models_dict : dict[str, Pipeline]
        Mapping of model name to fitted pipeline.

    Returns
    -------
    pd.DataFrame with one row per model and metric columns.
    """
    rows = []
    for name, pipe in models_dict.items():
        y_pred = pipe.predict(X_test)
        y_proba = None
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba=y_proba)
        metrics["model"] = name
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("model")


def plot_roc_curves(models_dict: dict, X_test, y_test, ax=None) -> Axes:
    """Plot ROC curves for multiple models on a single axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    for name, pipe in models_dict.items():
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            roc = compute_roc_data(y_test, y_proba)
            ax.plot(roc["fpr"], roc["tpr"], label=f"{name} (AUC={roc['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    return ax


def plot_confusion_matrix(y_true, y_pred, model_name: str = "", ax=None) -> Axes:
    """Plot confusion matrix."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    if model_name:
        ax.set_title(f"Confusion Matrix - {model_name}")
    return ax


def plot_feature_importances(pipeline, top_n: int = 15, ax=None) -> Axes:
    """Plot top-N feature importances from a fitted pipeline."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_).ravel()
    else:
        raise ValueError("Classifier has no feature_importances_ or coef_")

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    indices = np.argsort(importances)[-top_n:][::-1]

    ax.barh(
        [feature_names[i] for i in reversed(indices)],
        [importances[i] for i in reversed(indices)],
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    return ax
