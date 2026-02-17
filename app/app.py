"""Main Streamlit entry point for the Liver Disease Prediction App."""

import streamlit as st

st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from components.about import render_about_tab
from components.model_info import render_model_info_tab
from components.prediction import render_prediction_tab
from components.styles import inject_css

inject_css()

# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">🔬 Clinical Decision Support · ML-Powered</div>
        <div class="hero-title">Liver Disease Risk Assessment</div>
        <div class="hero-subtitle">
            Random Forest · F1 0.90 · AUC 0.95 · 1,700-record dataset · For educational use only
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_predict, tab_info, tab_about = st.tabs([
    "🩺  Prediction",
    "📊  Model Performance",
    "ℹ️  About",
])

with tab_predict:
    render_prediction_tab()

with tab_info:
    render_model_info_tab()

with tab_about:
    render_about_tab()
