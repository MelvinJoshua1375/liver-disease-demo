"""Prediction tab: patient input form + risk result display."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.components.utils import FEATURE_CONFIG, FEATURE_ORDER, load_model


def _risk_level(prob: float) -> tuple[str, str, str]:
    """Return (level_label, css_class, emoji) based on probability."""
    if prob < 0.35:
        return "Low Risk", "risk-low", "✅"
    elif prob < 0.65:
        return "Moderate Risk", "risk-medium", "⚠️"
    else:
        return "High Risk", "risk-high", "🔴"


def _feature_importance_chart(model, feature_names: list[str]) -> plt.Figure:
    """Mini horizontal bar chart of RF feature importances."""
    classifier = model.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "feature_importances_"):
        return None

    preprocessor = model.named_steps["preprocessor"]
    try:
        feat_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feat_names = [f"Feature {i}" for i in range(len(classifier.feature_importances_))]

    importances = classifier.feature_importances_
    idx = np.argsort(importances)[-10:]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(
        [feat_names[i] for i in idx],
        importances[idx],
        color="#4A90D9",
        edgecolor="white",
    )
    ax.set_xlabel("Importance", fontsize=8)
    ax.set_title("Top Feature Importances", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def render_prediction_tab() -> None:
    """Render the full Prediction tab UI."""
    model = load_model()

    if model is None:
        st.error(
            "Model file not found. Run `python scripts/train_model.py` first.",
            icon="⚠️",
        )
        return

    # ── Layout: input left, results right ───────────────────────────────────
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("Patient Information")

        st.markdown('<div class="section-header">Clinical Measurements</div>', unsafe_allow_html=True)
        num_col1, num_col2 = st.columns(2)
        inputs: dict = {}

        num_feats = list(FEATURE_CONFIG["numeric"].items())
        for idx, (feat, cfg) in enumerate(num_feats):
            col = num_col1 if idx % 2 == 0 else num_col2
            with col:
                inputs[feat] = st.number_input(
                    cfg["label"],
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(cfg["default"]),
                    step=float(cfg["step"]),
                    help=cfg.get("help", ""),
                    format="%.1f" if cfg["type"] is float else "%d",
                )

        st.markdown('<div class="section-header">Patient History</div>', unsafe_allow_html=True)
        cat_col1, cat_col2 = st.columns(2)
        cat_feats = list(FEATURE_CONFIG["categorical"].items())
        for idx, (feat, cfg) in enumerate(cat_feats):
            col = cat_col1 if idx % 2 == 0 else cat_col2
            with col:
                inputs[feat] = st.selectbox(
                    feat,
                    options=cfg["options"],
                    index=cfg["options"].index(cfg["default"]),
                )

        predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    with right_col:
        st.subheader("Prediction Result")

        if not predict_clicked:
            st.info("Fill in the patient information and click **Predict** to see the result.", icon="ℹ️")
            return

        # Build input DataFrame in exact feature order
        row = {feat: [inputs[feat]] for feat in FEATURE_ORDER}
        input_df = pd.DataFrame(row)

        with st.spinner("Calculating..."):
            model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

        level, css_class, emoji = _risk_level(prob)

        # Risk banner
        st.markdown(
            f'<div class="{css_class}"><strong>{emoji} {level}</strong><br/>'
            f'Probability of Liver Disease: <strong>{prob:.1%}</strong></div>',
            unsafe_allow_html=True,
        )

        # Probability gauge
        st.markdown("**Risk Probability**")
        st.progress(float(prob))
        gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
        gauge_col1.markdown("Low (<35%)")
        gauge_col2.markdown("<div style='text-align:center'>Moderate (35-65%)</div>", unsafe_allow_html=True)
        gauge_col3.markdown("<div style='text-align:right'>High (>65%)</div>", unsafe_allow_html=True)

        # Probability metrics
        m1, m2 = st.columns(2)
        m1.metric("Probability (Positive)", f"{prob:.3f}")
        m2.metric("Probability (Negative)", f"{1 - prob:.3f}")

        # Input summary
        with st.expander("Input Summary", expanded=False):
            summary_df = pd.DataFrame({
                "Feature": list(inputs.keys()),
                "Value": list(inputs.values()),
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Feature importances chart
        fig = _feature_importance_chart(model, FEATURE_ORDER)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.caption(
            "⚕️ This is a machine learning-based prediction and **not** a medical diagnosis. "
            "Always consult a qualified healthcare professional."
        )
