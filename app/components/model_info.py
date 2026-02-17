"""Model Performance tab: KPI metrics, comparison table, feature importances."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from components.utils import load_metadata

_MODEL_COMPARISON = [
    {"Model": "Logistic Regression",  "Accuracy": 0.821, "F1 (Weighted)": 0.821, "ROC AUC": 0.896},
    {"Model": "Decision Tree",        "Accuracy": 0.839, "F1 (Weighted)": 0.839, "ROC AUC": 0.900},
    {"Model": "SVM",                  "Accuracy": 0.850, "F1 (Weighted)": 0.850, "ROC AUC": 0.926},
    {"Model": "Naive Bayes",          "Accuracy": 0.780, "F1 (Weighted)": 0.780, "ROC AUC": 0.865},
    {"Model": "Random Forest ★",      "Accuracy": 0.897, "F1 (Weighted)": 0.897, "ROC AUC": 0.953},
    {"Model": "DT + LR Hybrid",       "Accuracy": 0.890, "F1 (Weighted)": 0.890, "ROC AUC": 0.960},
]


def render_model_info_tab() -> None:
    meta = load_metadata()

    # ── KPI headline metrics ───────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-bottom:0.5rem">
            <span style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                         letter-spacing:1.4px; color:#94A3B8">
                Deployed Model — Random Forest
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",       f"{meta.get('accuracy',  0.897):.1%}")
    c2.metric("F1 Score",       f"{meta.get('f1_score',  0.897):.3f}")
    c3.metric("ROC AUC",        f"{meta.get('roc_auc',   0.953):.3f}")
    c4.metric("Training Rows",  f"{meta.get('dataset_size', 1700):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        # Model comparison table
        st.markdown(
            '<div class="section-header">Model Comparison</div>',
            unsafe_allow_html=True,
        )
        cmp_df = pd.DataFrame(_MODEL_COMPARISON)
        styled = (
            cmp_df.style
            .highlight_max(subset=["Accuracy", "F1 (Weighted)", "ROC AUC"], color="#DBEAFE")
            .format({"Accuracy": "{:.3f}", "F1 (Weighted)": "{:.3f}", "ROC AUC": "{:.3f}"})
            .set_properties(**{"font-size": "13px"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=260)

        # Hyperparameters
        st.markdown(
            '<div class="section-header">Hyperparameters</div>',
            unsafe_allow_html=True,
        )
        hp = meta.get("hyperparameters", {
            "n_estimators": 100, "max_depth": 10,
            "min_samples_split": 10, "min_samples_leaf": 4,
        })

        hp_cols = st.columns(len(hp))
        labels = {
            "n_estimators": "Trees", "max_depth": "Max Depth",
            "min_samples_split": "Min Split", "min_samples_leaf": "Min Leaf",
        }
        for col, (k, v) in zip(hp_cols, hp.items(), strict=False):
            col.metric(labels.get(k, k), v)

    with right:
        st.markdown(
            '<div class="section-header">F1 Score by Model</div>',
            unsafe_allow_html=True,
        )
        _plot_comparison_chart()

        st.markdown(
            '<div class="section-header">Feature Importances</div>',
            unsafe_allow_html=True,
        )
        _plot_feature_importances(meta)


def _plot_comparison_chart() -> None:
    models = [r["Model"].replace(" ★", "") for r in _MODEL_COMPARISON]
    f1s    = [r["F1 (Weighted)"] for r in _MODEL_COMPARISON]
    colors = ["#93C5FD" if "★" not in r["Model"] and "Hybrid" not in r["Model"]
              else "#1D4ED8" for r in _MODEL_COMPARISON]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.barh(models, f1s, color=colors, edgecolor="white", height=0.55)
    for bar, val in zip(bars, f1s, strict=False):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color="#64748B")

    ax.set_xlim(0.7, 1.01)
    ax.axvline(0.9, color="#1D4ED8", linestyle="--", alpha=0.3, linewidth=1.2)
    ax.set_xlabel("F1 Score (Weighted)", fontsize=8.5, color="#64748B")
    ax.tick_params(labelsize=8, colors="#64748B")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    ax.grid(axis="x", alpha=0.15, linestyle="--", color="#E2E8F0")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _plot_feature_importances(meta: dict) -> None:
    fi = meta.get("feature_importances", {})
    if not fi:
        st.caption("Feature importances not available.")
        return

    feats = list(fi.keys())
    vals  = list(fi.values())
    idx   = list(reversed(sorted(range(len(vals)), key=lambda i: vals[i])))[:8]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sorted_feats = [feats[i] for i in idx]
    sorted_vals  = [vals[i]  for i in idx]
    colors = ["#1D4ED8" if v == max(sorted_vals) else
              "#3B82F6" if v >= sorted(sorted_vals)[-3] else
              "#93C5FD" for v in sorted_vals]

    bars = ax.barh(sorted_feats, sorted_vals, color=colors, edgecolor="white", height=0.55)
    for bar, val in zip(bars, sorted_vals, strict=False):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7.5, color="#64748B")

    ax.set_xlabel("Importance", fontsize=8.5, color="#64748B")
    ax.tick_params(labelsize=8, colors="#64748B")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    ax.grid(axis="x", alpha=0.15, linestyle="--", color="#E2E8F0")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
