"""WoE and IV visualization functions."""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.woe import calculate_iv, calculate_woe


def plot_woe_bars(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10,
    ax: plt.Axes | None = None,
    positive_label: str = "Positive",
    negative_label: str = "Negative",
) -> plt.Axes:
    """Bar chart of WoE values per bin for a single feature.

    Positive WoE (more non-events) → blue. Negative WoE (more events) → red.
    """
    _, detail = calculate_woe(df, feature, target, positive_label, negative_label, bins)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    labels = detail["bin"].astype(str).tolist()
    values = detail["woe"].tolist()
    colors = ["#4A90D9" if v >= 0 else "#E86C00" for v in values]

    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values, strict=False):
        offset = 0.02 if val >= 0 else -0.02
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:.3f}",
            ha="center", va=va, fontsize=8, color="#1A1A2E",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Weight of Evidence", fontsize=10)
    ax.set_title(f"WoE: {feature}", fontsize=11, fontweight="bold", pad=8)
    ax.axhline(0, color="#1A1A2E", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    return ax


def plot_iv_comparison(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    bins: int = 10,
    ax: plt.Axes | None = None,
    positive_label: str = "Positive",
    negative_label: str = "Negative",
) -> plt.Axes:
    """Horizontal bar chart comparing IV across all features.

    Bars are colored by IV interpretation threshold.
    """
    ivs = []
    for feat in features:
        iv_val = calculate_iv(df, feat, target, positive_label, negative_label, bins)
        ivs.append({"Feature": feat, "IV": iv_val})

    iv_df = pd.DataFrame(ivs).sort_values("IV", ascending=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(4, len(features) * 0.55)))

    def _color(iv: float) -> str:
        if iv < 0.02:
            return "#9E9E9E"
        elif iv < 0.1:
            return "#FFC107"
        elif iv < 0.3:
            return "#4CAF50"
        elif iv < 0.5:
            return "#4A90D9"
        return "#E86C00"

    colors = [_color(v) for v in iv_df["IV"]]
    bars = ax.barh(iv_df["Feature"], iv_df["IV"], color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, iv_df["IV"], strict=False):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=9, color="#1A1A2E",
        )

    thresholds = [(0.02, "Not Useful"), (0.1, "Weak"), (0.3, "Medium"), (0.5, "Strong")]
    for thr, label in thresholds:
        ax.axvline(thr, color="gray", linestyle="--", alpha=0.4, linewidth=0.9)
        ax.text(thr + 0.002, len(iv_df) - 0.3, label, fontsize=7, color="gray")

    ax.set_xlabel("Information Value", fontsize=10)
    ax.set_title("Information Value by Feature", fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    return ax


def validate_woe_monotonicity(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10,
    expected_direction: str | None = None,
    positive_label: str = "Positive",
    negative_label: str = "Negative",
    categories: list | None = None,
) -> dict:
    """Check whether WoE values are monotonic across ordered bins.

    Useful for ordinal features and binned numeric features.

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
    target : str
    bins : int
    expected_direction : "increasing", "decreasing", or None (auto-detect).
    positive_label, negative_label : str
    categories : list or None
        Explicit ordering of bin labels for ordinal features
        (e.g., ["Low", "Medium", "High"]). When provided, the WoE values
        are reordered to match this sequence before checking monotonicity.

    Returns
    -------
    dict with keys:
        is_monotonic : bool
        direction    : "increasing" | "decreasing" | "non-monotonic"
        violations   : list of (bin_i, bin_j, woe_i, woe_j) tuples
        woe_values   : list of (bin_label, woe) tuples
    """
    _, detail = calculate_woe(df, feature, target, positive_label, negative_label, bins)

    # Reorder rows by explicit category list when provided
    if categories is not None:
        cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
        detail["bin"] = detail["bin"].astype(str).astype(cat_type)
        detail = detail.sort_values("bin").reset_index(drop=True)

    woe_vals = detail["woe"].values
    bin_labels = detail["bin"].astype(str).values

    diffs = np.diff(woe_vals)
    all_up = bool(np.all(diffs >= 0))
    all_down = bool(np.all(diffs <= 0))

    direction = "increasing" if all_up else ("decreasing" if all_down else "non-monotonic")

    violations = []
    if not (all_up or all_down):
        for i, diff in enumerate(diffs):
            if expected_direction == "increasing" and diff < 0 or expected_direction == "decreasing" and diff > 0 or (
                expected_direction is None
                and abs(diff) > 1e-6
                and i > 0
                and np.sign(diffs[i]) != np.sign(diffs[i - 1])
                and diffs[i - 1] != 0
            ):
                violations.append((bin_labels[i], bin_labels[i + 1], woe_vals[i], woe_vals[i + 1]))

    return {
        "is_monotonic": all_up or all_down,
        "direction": direction,
        "violations": violations,
        "woe_values": list(zip(bin_labels, woe_vals.tolist(), strict=False)),
    }
