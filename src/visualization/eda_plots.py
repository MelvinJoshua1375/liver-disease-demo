"""EDA visualization helpers used in the narrative notebook."""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.visualization.style import PALETTE


def crosstabulate(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10,
) -> pd.DataFrame:
    """Bin feature into quantile groups and count target events per bin.

    For categorical features, groups by unique values.
    For numeric features, groups by pd.qcut quantiles.

    Returns
    -------
    pd.DataFrame
        Columns: bin, Positive, Negative, total, positive_pct, negative_pct.
    """
    data = df[[feature, target]].copy()

    if pd.api.types.is_numeric_dtype(data[feature]):
        data["bin"] = pd.qcut(data[feature], q=bins, duplicates="drop")
    else:
        data["bin"] = data[feature]

    ct = (
        data.groupby("bin", observed=True)[target]
        .value_counts()
        .unstack(fill_value=0)
    )

    ct.columns.name = None
    ct = ct.reset_index()

    pos = ct.get("Positive", 0)
    neg = ct.get("Negative", 0)
    ct["total"] = pos + neg
    ct["positive_pct"] = (pos / ct["total"] * 100).round(1)
    ct["negative_pct"] = (neg / ct["total"] * 100).round(1)

    return ct


def plot_percentage_stacked_chart(
    ct: pd.DataFrame,
    title: str = "",
    xlabel: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """100% stacked bar chart from a crosstabulate output DataFrame."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    bins_str = ct["bin"].astype(str).tolist()
    pos = ct.get("positive_pct", 0).tolist()
    neg = ct.get("negative_pct", 0).tolist()

    ax.bar(bins_str, pos, label="Positive", color=PALETTE["positive"], width=0.6)
    ax.bar(bins_str, neg, bottom=pos, label="Negative", color=PALETTE["negative"], width=0.6)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_xlabel(xlabel or "Bin", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(axis="x", rotation=40)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    return ax


def plot_boxplot(
    df: pd.DataFrame,
    feature: str,
    target: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Side-by-side boxplot comparing feature distribution by target class."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    groups = [df.loc[df[target] == lbl, feature].dropna() for lbl in ["Negative", "Positive"]]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                    medianprops={"color": "white", "linewidth": 2})

    colors = [PALETTE["negative"], PALETTE["positive"]]
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)

    t_stat, p_val = stats.ttest_ind(*groups, equal_var=False)
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))

    ax.set_xticklabels(["Negative", "Positive"], fontsize=10)
    ax.set_ylabel(feature, fontsize=10)
    ax.set_title(f"{feature} (p={p_val:.4f} {sig})", fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    return ax


def plot_correlation_matrix(
    df: pd.DataFrame,
    numeric_features: list[str],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Annotated heatmap of pairwise Pearson correlations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    corr = df[numeric_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9},
    )

    ax.set_title("Correlation Matrix (Numeric Features)", fontsize=11, fontweight="bold", pad=10)

    return ax
