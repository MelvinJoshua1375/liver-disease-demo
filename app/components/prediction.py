"""Prediction tab: patient presets, slider inputs, fragment-based results."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_lottie import st_lottie

from components.utils import FEATURE_CONFIG, FEATURE_ORDER, load_model

# ── Patient presets ──────────────────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    "Healthy 30yo": {
        "Age": 30, "BMI": 22.5, "AlcoholConsumption": 2.0,
        "PhysicalActivity": 7.0, "LiverFunctionTest": 35.0,
        "Gender": "Female", "Smoking": "No", "GeneticRisk": "Low",
        "Diabetes": "No", "Hypertension": "No",
    },
    "Average 45yo": {
        "Age": 45, "BMI": 27.0, "AlcoholConsumption": 8.5,
        "PhysicalActivity": 4.5, "LiverFunctionTest": 55.0,
        "Gender": "Male", "Smoking": "No", "GeneticRisk": "Medium",
        "Diabetes": "No", "Hypertension": "No",
    },
    "At-risk 55yo": {
        "Age": 55, "BMI": 32.0, "AlcoholConsumption": 14.0,
        "PhysicalActivity": 2.0, "LiverFunctionTest": 72.0,
        "Gender": "Male", "Smoking": "Yes", "GeneticRisk": "Medium",
        "Diabetes": "Yes", "Hypertension": "No",
    },
    "High-risk 65yo": {
        "Age": 65, "BMI": 38.0, "AlcoholConsumption": 18.0,
        "PhysicalActivity": 1.0, "LiverFunctionTest": 88.0,
        "Gender": "Male", "Smoking": "Yes", "GeneticRisk": "High",
        "Diabetes": "Yes", "Hypertension": "Yes",
    },
}

# Lottie JSON for medical stethoscope animation (inline, lightweight)
_LOTTIE_MEDICAL = {
    "v": "5.5.7", "fr": 30, "ip": 0, "op": 60, "w": 200, "h": 200,
    "assets": [],
    "layers": [{
        "ty": 4, "nm": "pulse", "sr": 1, "ks": {
            "o": {"a": 1, "k": [
                {"i": {"x": [0.4], "y": [1]}, "o": {"x": [0.6], "y": [0]}, "t": 0, "s": [40]},
                {"i": {"x": [0.4], "y": [1]}, "o": {"x": [0.6], "y": [0]}, "t": 30, "s": [100]},
                {"t": 60, "s": [40]},
            ]},
            "r": {"a": 0, "k": 0}, "p": {"a": 0, "k": [100, 100]},
            "a": {"a": 0, "k": [0, 0]},
            "s": {"a": 1, "k": [
                {"i": {"x": [0.4, 0.4, 0.4], "y": [1, 1, 1]}, "o": {"x": [0.6, 0.6, 0.6], "y": [0, 0, 0]}, "t": 0, "s": [90, 90, 100]},
                {"i": {"x": [0.4, 0.4, 0.4], "y": [1, 1, 1]}, "o": {"x": [0.6, 0.6, 0.6], "y": [0, 0, 0]}, "t": 30, "s": [110, 110, 100]},
                {"t": 60, "s": [90, 90, 100]},
            ]},
        },
        "shapes": [{
            "ty": "el", "p": {"a": 0, "k": [0, 0]},
            "s": {"a": 0, "k": [60, 60]},
        }, {
            "ty": "st", "c": {"a": 0, "k": [0.12, 0.31, 0.84, 1]},
            "o": {"a": 0, "k": 100}, "w": {"a": 0, "k": 3},
        }, {
            "ty": "fl", "c": {"a": 0, "k": [0.93, 0.95, 1, 1]},
            "o": {"a": 0, "k": 100},
        }],
        "ip": 0, "op": 60, "st": 0,
    }],
}


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
        x=vals, y=names, orientation="h",
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
        xaxis={"showgrid": True, "gridcolor": "#F1F5F9", "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "tickfont": {"size": 11, "color": "#475569"}},
        bargap=0.3,
    )
    return fig


def _apply_preset(preset_name: str) -> None:
    """Apply a patient preset — sets slider + selectbox keys directly."""
    vals = PRESETS[preset_name]
    for feat, val in vals.items():
        if feat in FEATURE_CONFIG["numeric"]:
            st.session_state[f"slider_{feat}"] = float(val)
        elif feat in FEATURE_CONFIG["categorical"]:
            st.session_state[f"select_{feat}"] = val


def render_prediction_tab() -> None:
    model = load_model()

    if model is None:
        st.error("Model not found. Run `python scripts/train_model.py` first.", icon="⚠️")
        return

    # ── Presets row ──────────────────────────────────────────────────────────
    st.markdown(
'<div class="section-header">Quick Presets</div>',
        unsafe_allow_html=True,
    )
    preset_cols = st.columns(len(PRESETS))
    for col, (name, _) in zip(preset_cols, PRESETS.items(), strict=False):
        with col:
            st.button(
                name,
                key=f"preset_{name}",
                on_click=_apply_preset,
                args=(name,),
                use_container_width=True,
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="large")

    # ── LEFT: Patient Input ──────────────────────────────────────────────────
    with left_col:
        st.markdown('<div class="card-title">Patient Profile</div>', unsafe_allow_html=True)

        st.markdown(
'<div class="section-header">Clinical Measurements</div>',
            unsafe_allow_html=True,
        )

        inputs: dict = {}
        for feat, cfg in FEATURE_CONFIG["numeric"].items():
            is_float = cfg["type"] is float
            slider_key = f"slider_{feat}"
            col_slider, col_val = st.columns([3, 1])
            with col_slider:
                # Only pass value if key is NOT already in session_state (avoids warning)
                slider_kwargs = {
                    "label": cfg["label"],
                    "min_value": float(cfg["min"]),
                    "max_value": float(cfg["max"]),
                    "step": float(cfg["step"]),
                    "help": cfg.get("help", ""),
                    "key": slider_key,
                }
                if slider_key not in st.session_state:
                    slider_kwargs["value"] = float(cfg["default"])
                slider_val = st.slider(**slider_kwargs)
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

        st.markdown(
'<div class="section-header">Medical History</div>',
            unsafe_allow_html=True,
        )
        cat_col1, cat_col2 = st.columns(2)
        cat_feats = list(FEATURE_CONFIG["categorical"].items())
        for i, (feat, cfg) in enumerate(cat_feats):
            col = cat_col1 if i % 2 == 0 else cat_col2
            with col:
                inputs[feat] = st.selectbox(
                    feat,
                    options=cfg["options"],
                    index=cfg["options"].index(cfg["default"]),
                    key=f"select_{feat}",
                )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button(
            "🔬  Run Risk Assessment",
            type="primary",
            use_container_width=True,
        )

    # ── RIGHT: Results (uses fragment for partial rerun) ──────────────────────
    with right_col:
        _render_results(model, inputs, predict_clicked)


@st.fragment
def _render_results(model, inputs: dict, predict_clicked: bool) -> None:
    """Result panel as a fragment — only this section reruns on predict."""
    st.markdown('<div class="card-title">Risk Assessment Result</div>', unsafe_allow_html=True)

    if not predict_clicked and "last_prob" not in st.session_state:
        # Empty state with lottie animation
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st_lottie(
            _LOTTIE_MEDICAL,
            height=120,
            key="lottie_empty",
        )
        st.markdown(
'<div class="empty-state" style="border:none; background:none; padding:0">'
'<div class="empty-state-title">Complete the patient profile</div>'
'<div class="empty-state-desc">Adjust the sliders and click <strong>Run Risk Assessment</strong></div>'
'</div>',
            unsafe_allow_html=True,
        )
        return

    if predict_clicked:
        row = {feat: [inputs[feat]] for feat in FEATURE_ORDER}
        input_df = pd.DataFrame(row)
        with st.spinner("Analysing patient data..."):
            prob = model.predict_proba(input_df)[0][1]
        st.session_state["last_prob"] = prob
        st.session_state["last_inputs"] = inputs.copy()
    else:
        prob = st.session_state["last_prob"]

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
        f'<div class="prob-grid">'
        f'<div class="prob-card">'
        f'<div class="prob-card-label">Positive</div>'
        f'<div class="prob-card-value {pos_color}">{prob:.1%}</div>'
        f'</div>'
        f'<div class="prob-card">'
        f'<div class="prob-card-label">Negative</div>'
        f'<div class="prob-card-value {neg_color}">{1-prob:.1%}</div>'
        f'</div>'
        f'</div>',
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
    display_inputs = st.session_state.get("last_inputs", inputs)
    with st.expander("📋  View Input Summary", expanded=False):
        summary_df = pd.DataFrame({
            "Feature": list(display_inputs.keys()),
            "Value": list(display_inputs.values()),
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown(
'<div class="disclaimer">'
'⚕️ This tool is for <strong>educational purposes only</strong> and is not a medical diagnosis. '
'Always consult a qualified healthcare professional.'
'</div>',
        unsafe_allow_html=True,
    )
