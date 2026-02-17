"""Prediction tab: patient input form + risk result display."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.utils import FEATURE_CONFIG, FEATURE_ORDER, load_model


def _risk_level(prob: float) -> tuple[str, str, str, str]:
    """Return (label, css_class, emoji, description) based on probability."""
    if prob < 0.35:
        return (
            "Low Risk", "risk-low", "&#x2705;",
            "No significant risk markers detected. Maintain healthy habits.",
        )
    elif prob < 0.65:
        return (
            "Moderate Risk", "risk-medium", "&#x26A0;&#xFE0F;",
            "Some risk factors present. Consider follow-up testing.",
        )
    else:
        return (
            "High Risk", "risk-high", "&#x1F534;",
            "Multiple risk factors detected. Consult a physician promptly.",
        )


def _gauge_chart(prob: float) -> go.Figure:
    """Plotly arc gauge showing probability 0-100%."""
    color = "#059669" if prob < 0.35 else ("#D97706" if prob < 0.65 else "#DC2626")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": "#1E3A5F", "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#E2E8F0",
                "tickfont": {"size": 10, "color": "#94A3B8", "family": "Inter"},
                "dtick": 20,
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "#ECFDF5"},
                {"range": [35, 65], "color": "#FFFBEB"},
                {"range": [65, 100], "color": "#FEF2F2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.82,
                "value": round(prob * 100, 1),
            },
        },
        title={
            "text": "Disease Probability",
            "font": {"size": 13, "color": "#64748B", "family": "Inter"},
        },
    ))
    fig.update_layout(
        height=240,
        margin={"t": 48, "b": 0, "l": 24, "r": 24},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def _feature_importance_chart(model) -> go.Figure | None:
    """Interactive Plotly horizontal bar chart of top-8 RF feature importances."""
    clf = model.named_steps.get("classifier")
    if clf is None or not hasattr(clf, "feature_importances_"):
        return None

    prep = model.named_steps["preprocessor"]
    try:
        feat_names = list(prep.get_feature_names_out())
    except Exception:
        feat_names = [f"Feature {i}" for i in range(len(clf.feature_importances_))]

    importances = clf.feature_importances_
    idx = np.argsort(importances)[-8:]

    names = [feat_names[i] for i in idx]
    vals = [importances[i] for i in idx]
    top3_cutoff = sorted(vals)[-3] if len(vals) >= 3 else 0
    colors = ["#1D4ED8" if v >= top3_cutoff else "#93C5FD" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker={"color": colors, "line": {"width": 0}},
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont={"size": 11, "color": "#64748B", "family": "Inter"},
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
            "gridwidth": 1,
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


def render_prediction_tab() -> None:
    model = load_model()

    if model is None:
        st.error("Model not found. Run `python scripts/train_model.py` first.", icon="⚠️")
        return

    left_col, right_col = st.columns([1, 1], gap="large")

    # ── LEFT: Patient Input ──────────────────────────────────────────────────
    with left_col:
        st.markdown('<div class="card-title">Patient Profile</div>', unsafe_allow_html=True)

        # -- Clinical Measurements (sliders + number inputs) --
        st.markdown('<div class="section-header">Clinical Measurements</div>', unsafe_allow_html=True)

        inputs: dict = {}
        for feat, cfg in FEATURE_CONFIG["numeric"].items():
            is_float = cfg["type"] is float
            col_slider, col_val = st.columns([3, 1])
            with col_slider:
                slider_val = st.slider(
                    cfg["label"],
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(cfg["default"]),
                    step=float(cfg["step"]),
                    help=cfg.get("help", ""),
                    key=f"slider_{feat}",
                )
            with col_val:
                num_val = st.number_input(
                    "Value",
                    min_value=float(cfg["min"]) if is_float else int(cfg["min"]),
                    max_value=float(cfg["max"]) if is_float else int(cfg["max"]),
                    value=int(slider_val) if not is_float else round(slider_val, 1),
                    step=float(cfg["step"]) if is_float else int(cfg["step"]),
                    format="%.1f" if is_float else "%d",
                    key=f"num_{feat}",
                    label_visibility="collapsed",
                )
            inputs[feat] = num_val

        # -- Medical History --
        st.markdown('<div class="section-header">Medical History</div>', unsafe_allow_html=True)
        cat_col1, cat_col2 = st.columns(2)
        cat_feats = list(FEATURE_CONFIG["categorical"].items())
        for i, (feat, cfg) in enumerate(cat_feats):
            col = cat_col1 if i % 2 == 0 else cat_col2
            with col:
                inputs[feat] = st.selectbox(
                    feat,
                    options=cfg["options"],
                    index=cfg["options"].index(cfg["default"]),
                )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button(
            "🔬  Run Risk Assessment",
            type="primary",
            use_container_width=True,
        )

    # ── RIGHT: Results ────────────────────────────────────────────────────────
    with right_col:
        st.markdown('<div class="card-title">Risk Assessment Result</div>', unsafe_allow_html=True)

        if not predict_clicked:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">🩺</div>
                    <div class="empty-state-title">
                        Complete the patient profile
                    </div>
                    <div class="empty-state-desc">
                        Adjust the sliders and click <strong>Run Risk Assessment</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        row = {feat: [inputs[feat]] for feat in FEATURE_ORDER}
        input_df = pd.DataFrame(row)

        with st.spinner("Analysing patient data..."):
            prob = model.predict_proba(input_df)[0][1]

        level, css_class, emoji, description = _risk_level(prob)

        # Risk banner
        st.markdown(
            f'<div class="{css_class}">'
            f'<div class="risk-label">{emoji} {level}</div>'
            f'<div class="risk-desc">{description}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Gauge chart
        st.plotly_chart(
            _gauge_chart(prob),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Probability cards
        pos_color = "positive" if prob >= 0.5 else ""
        neg_color = "negative" if prob < 0.5 else ""
        st.markdown(
            f"""
            <div class="prob-grid">
                <div class="prob-card">
                    <div class="prob-card-label">Positive</div>
                    <div class="prob-card-value {pos_color}">{prob:.1%}</div>
                </div>
                <div class="prob-card">
                    <div class="prob-card-label">Negative</div>
                    <div class="prob-card-value {neg_color}">{1-prob:.1%}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Feature importance (interactive Plotly)
        fig = _feature_importance_chart(model)
        if fig is not None:
            st.markdown(
                '<div class="section-header">Key Predictors</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Input summary
        with st.expander("📋  View Input Summary", expanded=False):
            summary_df = pd.DataFrame({
                "Feature": list(inputs.keys()),
                "Value": list(inputs.values()),
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="disclaimer">'
            '⚕️ This tool is for <strong>educational purposes only</strong> and is not a medical diagnosis. '
            'Always consult a qualified healthcare professional.'
            '</div>',
            unsafe_allow_html=True,
        )
