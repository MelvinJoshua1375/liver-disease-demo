"""Model Performance tab: KPI metrics, ROC curves, confusion matrix, comparisons, SHAP."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from components.utils import load_metadata, load_model

_MODEL_COMPARISON = [
    {"Model": "Logistic Regression", "Accuracy": 0.821, "F1 (Weighted)": 0.821, "ROC AUC": 0.896},
    {"Model": "Decision Tree", "Accuracy": 0.839, "F1 (Weighted)": 0.839, "ROC AUC": 0.900},
    {"Model": "SVM", "Accuracy": 0.850, "F1 (Weighted)": 0.850, "ROC AUC": 0.926},
    {"Model": "Naive Bayes", "Accuracy": 0.780, "F1 (Weighted)": 0.780, "ROC AUC": 0.865},
    {"Model": "Random Forest", "Accuracy": 0.897, "F1 (Weighted)": 0.897, "ROC AUC": 0.953},
    {"Model": "DT + LR Hybrid", "Accuracy": 0.890, "F1 (Weighted)": 0.890, "ROC AUC": 0.960},
]

_ROC_COLORS = {
    "Logistic Regression": "#94A3B8",
    "Decision Tree": "#CBD5E1",
    "Svc": "#64748B",
    "Naive Bayes": "#E2E8F0",
    "Random Forest": "#1D4ED8",
}


def _roc_chart(meta: dict) -> go.Figure | None:
    """Interactive multi-model ROC curve plot."""
    roc_data = meta.get("roc_curves", {})
    if not roc_data:
        return None

    fig = go.Figure()

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line={"color": "#E2E8F0", "width": 1, "dash": "dash"},
        showlegend=False,
        hoverinfo="skip",
    ))

    for model_name, data in roc_data.items():
        color = _ROC_COLORS.get(model_name, "#94A3B8")
        width = 3 if model_name == "Random Forest" else 1.5
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"],
            mode="lines",
            name=f"{model_name} (AUC={data['auc']:.3f})",
            line={"color": color, "width": width},
            hovertemplate=f"<b>{model_name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        height=380,
        margin={"t": 16, "b": 48, "l": 48, "r": 16},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "size": 11},
        legend={
            "orientation": "h",
            "yanchor": "top", "y": -0.15,
            "xanchor": "center", "x": 0.5,
            "font": {"size": 10},
        },
        xaxis={
            "title": "False Positive Rate",
            "titlefont": {"size": 11, "color": "#64748B"},
            "showgrid": True, "gridcolor": "#F1F5F9",
            "zeroline": False,
            "tickfont": {"color": "#94A3B8"},
        },
        yaxis={
            "title": "True Positive Rate",
            "titlefont": {"size": 11, "color": "#64748B"},
            "showgrid": True, "gridcolor": "#F1F5F9",
            "zeroline": False,
            "tickfont": {"color": "#94A3B8"},
        },
    )
    return fig


def _confusion_matrix_chart(meta: dict) -> go.Figure | None:
    """Plotly heatmap confusion matrix."""
    cm = meta.get("confusion_matrix")
    if not cm:
        return None

    labels = ["Negative", "Positive"]
    # Annotations with counts and percentages
    total = sum(sum(row) for row in cm)
    text = [[f"{cm[i][j]}<br>({cm[i][j]/total:.1%})" for j in range(2)] for i in range(2)]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels, y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14, "family": "Inter"},
        colorscale=[[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, "#1D4ED8"]],
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        height=320,
        margin={"t": 16, "b": 48, "l": 60, "r": 16},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        xaxis={"title": "Predicted", "titlefont": {"size": 11, "color": "#64748B"}, "tickfont": {"size": 11}},
        yaxis={"title": "Actual", "titlefont": {"size": 11, "color": "#64748B"}, "tickfont": {"size": 11}, "autorange": "reversed"},
    )
    return fig


def _comparison_chart() -> go.Figure:
    """Interactive grouped bar chart comparing model metrics."""
    df = pd.DataFrame(_MODEL_COMPARISON)
    colors = {"Accuracy": "#93C5FD", "F1 (Weighted)": "#3B82F6", "ROC AUC": "#1D4ED8"}

    fig = go.Figure()
    for metric, color in colors.items():
        fig.add_trace(go.Bar(
            name=metric,
            y=df["Model"], x=df[metric],
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
            "showgrid": True, "gridcolor": "#F1F5F9",
            "zeroline": False, "tickformat": ".2f",
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
        x=vals, y=names, orientation="h",
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
        xaxis={"showgrid": True, "gridcolor": "#F1F5F9", "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "tickfont": {"size": 11, "color": "#475569"}},
        bargap=0.3,
    )
    return fig


@st.cache_data(show_spinner="Computing SHAP values...")
def _compute_global_shap():
    """Compute global SHAP importances using a sample of training data."""
    try:
        from src.config import load_settings
        from src.data.loader import load_raw_data
        from src.evaluation.shap_explain import aggregate_shap_importances, compute_shap_values
        from src.features.schema import ALL_FEATURES

        model = load_model()
        if model is None:
            return None

        settings = load_settings()
        df = load_raw_data(settings)
        X = df[ALL_FEATURES].sample(n=min(300, len(df)), random_state=42)
        shap_exp = compute_shap_values(model, X)
        return aggregate_shap_importances(shap_exp)
    except Exception:
        return None


def _shap_global_chart(shap_df: pd.DataFrame) -> go.Figure:
    """Interactive Plotly bar chart of mean |SHAP| importances."""
    df = shap_df.sort_values("mean_abs_shap", ascending=True).tail(10)
    top3_cutoff = df["mean_abs_shap"].nlargest(3).min()
    colors = ["#7C3AED" if v >= top3_cutoff else "#C4B5FD" for v in df["mean_abs_shap"]]

    fig = go.Figure(go.Bar(
        x=df["mean_abs_shap"].values,
        y=df["feature"].values,
        orientation="h",
        marker={"color": colors, "line": {"width": 0}},
        text=[f"{v:.3f}" for v in df["mean_abs_shap"]],
        textposition="outside",
        textfont={"size": 10, "color": "#64748B"},
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        margin={"t": 8, "b": 8, "l": 8, "r": 48},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        xaxis={"showgrid": True, "gridcolor": "#F1F5F9", "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "tickfont": {"size": 11, "color": "#475569"}},
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

    # ── ROC Curve + Confusion Matrix row ──────────────────────────────────────
    roc_col, cm_col = st.columns([1.2, 1], gap="large")

    with roc_col:
        st.markdown(
'<div class="section-header">ROC Curves (All Models)</div>',
            unsafe_allow_html=True,
        )
        roc_fig = _roc_chart(meta)
        if roc_fig:
            st.plotly_chart(roc_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("ROC curve data not available. Regenerate metadata.")

    with cm_col:
        st.markdown(
'<div class="section-header">Confusion Matrix</div>',
            unsafe_allow_html=True,
        )
        cm_fig = _confusion_matrix_chart(meta)
        if cm_fig:
            st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Confusion matrix not available. Regenerate metadata.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Model comparison + Feature importances / Hyperparams ──────────────────
    left, right = st.columns([1.1, 1], gap="large")

    with left:
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
            st.plotly_chart(fi_chart, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Feature importances not available.")

    # ── SHAP Global Importance ───────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    shap_left, shap_right = st.columns([1.1, 1], gap="large")
    with shap_left:
        st.markdown(
'<div class="section-header">SHAP Global Feature Importance</div>',
            unsafe_allow_html=True,
        )
        shap_df = _compute_global_shap()
        if shap_df is not None:
            st.plotly_chart(
                _shap_global_chart(shap_df),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption("Mean |SHAP value| across 300 sampled patients — measures each feature's average impact on predictions")
        else:
            st.caption("SHAP values not available. Ensure model and data are present.")

    with shap_right:
        st.markdown(
'<div class="section-header">Understanding SHAP</div>',
            unsafe_allow_html=True,
        )
        st.markdown("""
**SHAP (SHapley Additive exPlanations)** decomposes each prediction into
per-feature contributions.

- **Global view** (left): average impact across many patients — which features matter most overall
- **Per-prediction view** (Prediction tab): why *this specific patient* got their risk score

Unlike standard feature importances (which measure split quality in trees),
SHAP values are grounded in game theory and provide **consistent, locally accurate** explanations.

> Higher mean |SHAP| = the feature moves predictions further from the baseline on average.
"""
        )
