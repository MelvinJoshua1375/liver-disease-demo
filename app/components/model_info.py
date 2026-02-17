"""Model Performance tab: KPI metrics, comparison table, feature importances."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.utils import load_metadata

_MODEL_COMPARISON = [
    {"Model": "Logistic Regression", "Accuracy": 0.821, "F1 (Weighted)": 0.821, "ROC AUC": 0.896},
    {"Model": "Decision Tree", "Accuracy": 0.839, "F1 (Weighted)": 0.839, "ROC AUC": 0.900},
    {"Model": "SVM", "Accuracy": 0.850, "F1 (Weighted)": 0.850, "ROC AUC": 0.926},
    {"Model": "Naive Bayes", "Accuracy": 0.780, "F1 (Weighted)": 0.780, "ROC AUC": 0.865},
    {"Model": "Random Forest", "Accuracy": 0.897, "F1 (Weighted)": 0.897, "ROC AUC": 0.953},
    {"Model": "DT + LR Hybrid", "Accuracy": 0.890, "F1 (Weighted)": 0.890, "ROC AUC": 0.960},
]


def _comparison_chart() -> go.Figure:
    """Interactive grouped bar chart comparing model metrics."""
    df = pd.DataFrame(_MODEL_COMPARISON)
    colors = {"Accuracy": "#93C5FD", "F1 (Weighted)": "#3B82F6", "ROC AUC": "#1D4ED8"}

    fig = go.Figure()
    for metric, color in colors.items():
        fig.add_trace(go.Bar(
            name=metric,
            y=df["Model"],
            x=df[metric],
            orientation="h",
            marker={"color": color, "line": {"width": 0}},
            hovertemplate="<b>%{y}</b><br>" + metric + ": %{x:.3f}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        height=300,
        margin={"t": 8, "b": 8, "l": 8, "r": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "size": 11},
        legend={
            "orientation": "h",
            "yanchor": "bottom", "y": 1.02,
            "xanchor": "right", "x": 1,
            "font": {"size": 10},
        },
        xaxis={
            "range": [0.7, 1.0],
            "showgrid": True,
            "gridcolor": "#F1F5F9",
            "zeroline": False,
            "tickformat": ".2f",
            "tickfont": {"color": "#94A3B8"},
        },
        yaxis={
            "showgrid": False,
            "tickfont": {"color": "#475569", "size": 11},
            "autorange": "reversed",
        },
        bargap=0.25,
        bargroupgap=0.08,
    )
    return fig


def _feature_importance_chart(meta: dict) -> go.Figure | None:
    """Interactive Plotly bar chart from metadata."""
    fi = meta.get("feature_importances", {})
    if not fi:
        return None

    sorted_items = sorted(fi.items(), key=lambda x: x[1])[-8:]
    names = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    top3_cutoff = sorted(vals)[-3] if len(vals) >= 3 else 0
    colors = ["#1D4ED8" if v >= top3_cutoff else "#93C5FD" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker={"color": colors, "line": {"width": 0}},
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont={"size": 10, "color": "#64748B"},
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        margin={"t": 8, "b": 8, "l": 8, "r": 48},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        xaxis={
            "showgrid": True,
            "gridcolor": "#F1F5F9",
            "zeroline": False,
            "showticklabels": False,
        },
        yaxis={
            "showgrid": False,
            "tickfont": {"size": 11, "color": "#475569"},
        },
        bargap=0.3,
    )
    return fig


def render_model_info_tab() -> None:
    meta = load_metadata()

    # ── KPI headline metrics ─────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Deployed Model &mdash; Random Forest</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{meta.get('accuracy',  0.897):.1%}")
    c2.metric("F1 Score", f"{meta.get('f1_score',  0.897):.3f}")
    c3.metric("ROC AUC", f"{meta.get('roc_auc',   0.953):.3f}")
    c4.metric("Training Rows", f"{meta.get('dataset_size', 1700):,}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        # Model comparison chart (interactive)
        st.markdown(
            '<div class="section-header">Model Comparison</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _comparison_chart(),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Comparison table
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        cmp_df = pd.DataFrame(_MODEL_COMPARISON)
        styled = (
            cmp_df.style
            .highlight_max(subset=["Accuracy", "F1 (Weighted)", "ROC AUC"], color="#DBEAFE")
            .format({"Accuracy": "{:.3f}", "F1 (Weighted)": "{:.3f}", "ROC AUC": "{:.3f}"})
            .set_properties(**{"font-size": "13px"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=260)

    with right:
        # Hyperparameters
        st.markdown(
            '<div class="section-header">Hyperparameters</div>',
            unsafe_allow_html=True,
        )
        hp = meta.get("hyperparameters", {
            "n_estimators": 100, "max_depth": 10,
            "min_samples_split": 10, "min_samples_leaf": 4,
        })

        hp_cols = st.columns(2)
        labels = {
            "n_estimators": "Trees", "max_depth": "Max Depth",
            "min_samples_split": "Min Split", "min_samples_leaf": "Min Leaf",
        }
        for i, (k, v) in enumerate(hp.items()):
            hp_cols[i % 2].metric(labels.get(k, k), v)

        # Feature importances (interactive)
        st.markdown(
            '<div class="section-header">Feature Importances</div>',
            unsafe_allow_html=True,
        )
        fi_chart = _feature_importance_chart(meta)
        if fi_chart is not None:
            st.plotly_chart(
                fi_chart,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.caption("Feature importances not available.")
