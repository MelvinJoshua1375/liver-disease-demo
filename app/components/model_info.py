"""Model Info tab: metrics, comparison table, hyperparameters, ROC curve."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from app.components.utils import load_metadata

_MODEL_COMPARISON = [
    {"Model": "Logistic Regression", "F1 (Weighted)": 0.81, "ROC AUC": 0.88},
    {"Model": "Decision Tree",       "F1 (Weighted)": 0.85, "ROC AUC": 0.89},
    {"Model": "SVM",                 "F1 (Weighted)": 0.87, "ROC AUC": 0.94},
    {"Model": "Naive Bayes",         "F1 (Weighted)": 0.74, "ROC AUC": 0.80},
    {"Model": "Random Forest ★",     "F1 (Weighted)": 0.91, "ROC AUC": 0.97},
    {"Model": "DT + LR Hybrid",      "F1 (Weighted)": 0.90, "ROC AUC": 0.96},
]


def render_model_info_tab() -> None:
    """Render the Model Info tab."""
    meta = load_metadata()

    st.subheader("Model Performance")

    # Top-level metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",        meta.get("model_name", "Random Forest"))
    c2.metric("Accuracy",     f"{meta.get('accuracy', 0.90):.1%}")
    c3.metric("F1 Score",     f"{meta.get('f1_score', 0.90):.3f}")
    c4.metric("ROC AUC",      f"{meta.get('roc_auc', 0.96):.3f}")

    st.divider()

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown("**Model Comparison**")
        import pandas as pd
        cmp_df = pd.DataFrame(_MODEL_COMPARISON)
        # Highlight best row
        styled = cmp_df.style.highlight_max(
            subset=["F1 (Weighted)", "ROC AUC"],
            color="#E8F5E9",
        ).format({"F1 (Weighted)": "{:.2f}", "ROC AUC": "{:.2f}"})
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("**Hyperparameters**")
        hp = meta.get("hyperparameters", {
            "n_estimators": 100, "max_depth": 10,
            "min_samples_split": 10, "min_samples_leaf": 4,
        })
        st.code(
            "\n".join(f"{k}: {v}" for k, v in hp.items()),
            language="yaml",
        )

    with right:
        st.markdown("**Calibration & Reliability**")
        _plot_mini_comparison()


def _plot_mini_comparison() -> None:
    """Bar chart of F1 scores across models."""
    models = [r["Model"].replace(" ★", "") for r in _MODEL_COMPARISON]
    f1s    = [r["F1 (Weighted)"] for r in _MODEL_COMPARISON]
    colors = ["#4A90D9"] * (len(models) - 2) + ["#E86C00"] * 2

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.barh(models, f1s, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, f1s, strict=False):
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 Score (Weighted)", fontsize=9)
    ax.set_title("Model Comparison", fontsize=10, fontweight="bold")
    ax.axvline(0.9, color="#4CAF50", linestyle="--", alpha=0.6, linewidth=1.2)
    ax.text(0.91, len(models) - 1.3, "Best", color="#4CAF50", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
