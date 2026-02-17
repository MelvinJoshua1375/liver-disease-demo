"""Probability calibration utilities."""

from matplotlib.axes import Axes
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline

from src.models.train import build_pipeline


def plot_calibration_curves(models_dict: dict, X_test, y_test, ax=None) -> Axes:
    """Plot calibration curves for multiple models."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    for name, pipe in models_dict.items():
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            fraction_pos, mean_predicted = calibration_curve(
                y_test, y_proba, n_bins=10
            )
            ax.plot(mean_predicted, fraction_pos, "s-", label=name)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend()
    return ax


def build_calibrated_pipeline(
    model_name: str,
    params: dict | None = None,
    method: str = "sigmoid",
) -> Pipeline:
    """Build a pipeline with a calibrated classifier."""
    return build_pipeline(model_name, params=params, calibrate=True)
