"""Prediction tab: patient input form + risk result display."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.utils import FEATURE_CONFIG, FEATURE_ORDER, load_model


def _risk_level(prob: float) -> tuple[str, str, str, str]:
    """Return (label, css_class, emoji, description) based on probability."""
    if prob < 0.35:
        return "Low Risk", "risk-low", "✅", "No significant markers detected."
    elif prob < 0.65:
        return "Moderate Risk", "risk-medium", "⚠️", "Some risk factors present. Consider follow-up."
    else:
        return "High Risk", "risk-high", "🔴", "Multiple risk factors detected. Consult a physician."


def _gauge_chart(prob: float) -> go.Figure:
    """Plotly arc gauge showing probability 0–100%."""
    color = "#059669" if prob < 0.35 else ("#D97706" if prob < 0.65 else "#DC2626")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": "#1E3A5F", "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#CBD5E1",
                "tickfont": {"size": 10, "color": "#94A3B8"},
            },
            "bar": {"color": color, "thickness": 0.26},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "#ECFDF5"},
                {"range": [35, 65], "color": "#FFFBEB"},
                {"range": [65, 100],"color": "#FEF2F2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": round(prob * 100, 1),
            },
        },
        title={
            "text": "Disease Probability",
            "font": {"size": 13, "color": "#64748B", "family": "Inter"},
        },
    ))
    fig.update_layout(
        height=260,
        margin={"t": 50, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="white",
        font={"family": "Inter"},
    )
    return fig


def _feature_importance_chart(model) -> plt.Figure | None:
    """Compact horizontal bar chart of top-8 RF feature importances."""
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

    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = ["#1D4ED8" if importances[i] >= sorted(importances)[-3] else "#93C5FD" for i in idx]
    bars = ax.barh([feat_names[i] for i in idx], importances[idx],
                   color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, importances[idx], strict=False):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7.5, color="#64748B")

    ax.set_xlabel("Importance Score", fontsize=8.5, color="#64748B")
    ax.set_title("Feature Importances", fontsize=10, fontweight="bold", color="#1E3A5F", pad=8)
    ax.tick_params(labelsize=8, colors="#64748B")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    ax.set_xlim(0, max(importances[idx]) * 1.25)
    ax.grid(axis="x", alpha=0.2, linestyle="--", color="#E2E8F0")
    fig.tight_layout()
    return fig


def render_prediction_tab() -> None:
    model = load_model()

    if model is None:
        st.error("Model not found. Run `python scripts/train_model.py` first.", icon="⚠️")
        return

    left_col, right_col = st.columns([1, 1], gap="large")

    # ── LEFT: Patient Input ────────────────────────────────────────────────────
    with left_col:
        st.markdown('<div class="card-title">Patient Profile</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Clinical Measurements</div>', unsafe_allow_html=True)
        num_col1, num_col2 = st.columns(2)
        inputs: dict = {}

        num_feats = list(FEATURE_CONFIG["numeric"].items())
        for i, (feat, cfg) in enumerate(num_feats):
            col = num_col1 if i % 2 == 0 else num_col2
            is_float = cfg["type"] is float
            with col:
                inputs[feat] = st.number_input(
                    cfg["label"],
                    min_value=float(cfg["min"]) if is_float else int(cfg["min"]),
                    max_value=float(cfg["max"]) if is_float else int(cfg["max"]),
                    value=float(cfg["default"]) if is_float else int(cfg["default"]),
                    step=float(cfg["step"]) if is_float else int(cfg["step"]),
                    help=cfg.get("help", ""),
                    format="%.1f" if is_float else "%d",
                )

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

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("Run Risk Assessment", type="primary", use_container_width=True)

    # ── RIGHT: Results ─────────────────────────────────────────────────────────
    with right_col:
        st.markdown('<div class="card-title">Risk Assessment Result</div>', unsafe_allow_html=True)

        if not predict_clicked:
            st.markdown(
                """
                <div style="
                    text-align:center; padding:48px 24px;
                    background:#F8FAFC; border-radius:16px;
                    border:2px dashed #CBD5E1; color:#94A3B8;
                ">
                    <div style="font-size:2.5rem; margin-bottom:12px">🩺</div>
                    <div style="font-weight:600; font-size:1rem; color:#64748B">
                        Complete the patient profile
                    </div>
                    <div style="font-size:0.85rem; margin-top:6px">
                        and click <strong>Run Risk Assessment</strong>
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
            f'<div class="risk-prob">{description}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge
        st.plotly_chart(_gauge_chart(prob), use_container_width=True, config={"displayModeBar": False})

        # Probability metrics
        m1, m2 = st.columns(2)
        m1.metric("Probability (Positive)", f"{prob:.1%}")
        m2.metric("Probability (Negative)", f"{1 - prob:.1%}")

        # Feature importance chart
        fig = _feature_importance_chart(model)
        if fig is not None:
            st.markdown('<div class="section-header">Key Predictors</div>', unsafe_allow_html=True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Input summary
        with st.expander("View Input Summary", expanded=False):
            summary_df = pd.DataFrame({
                "Feature": list(inputs.keys()),
                "Value": list(inputs.values()),
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.caption(
            "⚕️ This tool is for educational purposes only and is **not** a medical diagnosis. "
            "Always consult a qualified healthcare professional."
        )
