"""Main Streamlit entry point for the Liver Disease Prediction App."""

import streamlit as st

st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from app.components.about import render_about_tab
from app.components.model_info import render_model_info_tab
from app.components.prediction import render_prediction_tab
from app.components.styles import inject_css

inject_css()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 Liver Disease Risk Prediction")
st.markdown(
    "A machine learning tool for assessing liver disease risk. "
    "Enter patient information in the **Prediction** tab to generate a risk estimate."
)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_info, tab_about = st.tabs([
    "🩺 Prediction",
    "📊 Model Info",
    "ℹ️ About",
])

with tab_predict:
    render_prediction_tab()

with tab_info:
    render_model_info_tab()

with tab_about:
    render_about_tab()
