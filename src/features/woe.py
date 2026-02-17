"""Weight of Evidence (WoE) and Information Value (IV) calculations.

Fixes over the original notebook implementation:
1. Smoothing applied ONLY when at least one bin has a zero count (not unconditionally).
2. Support for numeric features via quantile binning (pd.qcut).
3. compute_woe_mappings / apply_woe_mappings pattern mirrors sklearn fit/transform,
   preventing data leakage in train/test splits.
"""


import numpy as np
import pandas as pd

# ─── Core WoE computation ────────────────────────────────────────────────────

def calculate_woe(
    df: pd.DataFrame,
    feature: str,
    target: str,
    positive_label: str = "Positive",
    negative_label: str = "Negative",
    bins: int = 10,
    smoothing: float = 0.5,
) -> tuple[dict, pd.DataFrame]:
    """Calculate Weight of Evidence for a single feature.

    Parameters
    ----------
    df : pd.DataFrame
        Training data only (never pass full dataset).
    feature : str
        Column name of the feature.
    target : str
        Column name of the binary target.
    positive_label, negative_label : str
        Target class labels.
    bins : int
        Number of quantile bins for numeric features.
    smoothing : float
        Laplace smoothing value applied ONLY when at least one bin has
        zero events or zero non-events.

    Returns
    -------
    woe_mapping : dict
        {bin_label: woe_value}
    detail_df : pd.DataFrame
        Columns: bin, events, non_events, dist_events, dist_non_events,
        woe, iv_component.
    """
    data = df[[feature, target]].copy()

    # Bin numeric features; leave categoricals as-is
    if pd.api.types.is_numeric_dtype(data[feature]):
        data["bin"] = pd.qcut(data[feature], q=bins, duplicates="drop")
    else:
        data["bin"] = data[feature]

    grouped = (
        data.groupby("bin", observed=True)
        .agg(
            events=(target, lambda x: (x == positive_label).sum()),
            non_events=(target, lambda x: (x == negative_label).sum()),
        )
        .reset_index()
    )

    # Apply Laplace smoothing ONLY if any bin has a zero count
    has_zero = (grouped["events"] == 0) | (grouped["non_events"] == 0)
    if has_zero.any():
        events_adj = grouped["events"] + smoothing
        non_events_adj = grouped["non_events"] + smoothing
    else:
        events_adj = grouped["events"].astype(float)
        non_events_adj = grouped["non_events"].astype(float)

    grouped["dist_events"] = events_adj / events_adj.sum()
    grouped["dist_non_events"] = non_events_adj / non_events_adj.sum()

    grouped["woe"] = np.log(grouped["dist_non_events"] / grouped["dist_events"])
    grouped["iv_component"] = (grouped["dist_non_events"] - grouped["dist_events"]) * grouped["woe"]

    woe_mapping = grouped.set_index("bin")["woe"].to_dict()
    return woe_mapping, grouped


def calculate_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    positive_label: str = "Positive",
    negative_label: str = "Negative",
    bins: int = 10,
    smoothing: float = 0.5,
) -> float:
    """Calculate total Information Value for a feature.

    IV interpretation: <0.02 not useful, 0.02-0.1 weak, 0.1-0.3 medium,
    0.3-0.5 strong, >0.5 suspicious.
    """
    _, detail = calculate_woe(df, feature, target, positive_label, negative_label, bins, smoothing)
    return float(detail["iv_component"].sum())


def iv_summary_table(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    bins: int = 10,
    positive_label: str = "Positive",
    negative_label: str = "Negative",
) -> pd.DataFrame:
    """Build IV summary table with interpretations for all features.

    Returns
    -------
    pd.DataFrame
        Columns: Feature, Total_IV, Interpretation. Sorted by IV descending.
    """
    rows = []
    for feat in features:
        iv = calculate_iv(df, feat, target, positive_label, negative_label, bins)
        if iv < 0.02:
            interp = "Not Useful"
        elif iv < 0.1:
            interp = "Weak"
        elif iv < 0.3:
            interp = "Medium"
        elif iv < 0.5:
            interp = "Strong"
        else:
            interp = "Suspicious"
        rows.append({"Feature": feat, "Total_IV": round(iv, 6), "Interpretation": interp})

    return (
        pd.DataFrame(rows)
        .sort_values("Total_IV", ascending=False)
        .reset_index(drop=True)
    )


# ─── Train/test safe WoE replacement ─────────────────────────────────────────

def compute_woe_mappings(
    train_df: pd.DataFrame,
    features: list[str],
    target: str,
    bins: int = 10,
    positive_label: str = "Positive",
    negative_label: str = "Negative",
) -> dict[str, dict]:
    """Learn WoE mappings from training data (fit step).

    For numeric features, stores both the WoE mapping and the bin edges
    derived from training data, so the same edges can be applied to test data
    via ``apply_woe_mappings``.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data ONLY. Never pass the full dataset.
    features : list[str]
        Columns to compute WoE for.
    target : str
    bins : int
    positive_label, negative_label : str

    Returns
    -------
    dict
        {feature_name: {"woe": {bin_label: woe_value}, "bin_edges": list | None}}
    """
    mappings = {}
    for feat in features:
        is_numeric = pd.api.types.is_numeric_dtype(train_df[feat])
        mapping, _ = calculate_woe(
            train_df, feat, target, positive_label, negative_label, bins
        )
        bin_edges = None
        if is_numeric:
            # Store the bin edges from training data for consistent binning
            _, bin_edges = pd.qcut(
                train_df[feat], q=bins, duplicates="drop", retbins=True
            )
            bin_edges = bin_edges.tolist()
        mappings[feat] = {"woe": mapping, "bin_edges": bin_edges}
    return mappings


def apply_woe_mappings(
    df: pd.DataFrame,
    mappings: dict[str, dict],
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Apply pre-learned WoE mappings to a DataFrame (transform step).

    For numeric features, uses the bin edges stored during
    ``compute_woe_mappings`` (via ``pd.cut``) so that train and test data
    are binned identically.  Unseen categories or out-of-range values are
    filled with *fill_value*.

    Parameters
    ----------
    df : pd.DataFrame
    mappings : dict
        Output of compute_woe_mappings.
    fill_value : float
        WoE for unseen categories. Default 0.0 (neutral log-odds).

    Returns
    -------
    pd.DataFrame
        Copy of df with WoE values replacing original feature columns.
    """
    result = df.copy()
    for feat, info in mappings.items():
        if feat not in result.columns:
            continue
        woe_map = info["woe"]
        bin_edges = info["bin_edges"]

        if bin_edges is not None:
            # Numeric: use training bin edges via pd.cut for consistency
            bins_col = pd.cut(result[feat], bins=bin_edges, include_lowest=True)
            # Map intervals to WoE values; convert to float directly
            # (unmapped intervals become NaN, then fill)
            mapped = bins_col.map(woe_map)
            result[feat] = pd.to_numeric(mapped, errors="coerce").fillna(fill_value)
        else:
            # Categorical: direct map
            result[feat] = (
                result[feat]
                .map(woe_map)
                .fillna(fill_value)
                .astype(float)
            )
    return result
