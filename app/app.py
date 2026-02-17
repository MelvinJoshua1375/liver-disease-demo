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

# ── Dark mode toggle ──────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode
theme_attr = 'data-theme="dark"' if dark else ""
icon = "☀️" if dark else "🌙"

# Inject JS to toggle the data-theme attribute on .stApp
st.markdown(
    f"""
    <script>
        const app = window.parent.document.querySelector('.stApp');
        if (app) {{ app.setAttribute('data-theme', '{"dark" if dark else ""}'); }}
    </script>
    <div class="theme-toggle" onclick="
        const app = window.parent.document.querySelector('.stApp');
        const current = app.getAttribute('data-theme');
        app.setAttribute('data-theme', current === 'dark' ? '' : 'dark');
    ">{icon}</div>
    """,
    unsafe_allow_html=True,
)

# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown(
'<div class="hero">'
'<div class="hero-badge">Clinical Decision Support &middot; ML-Powered</div>'
'<div class="hero-title">Liver Disease Risk Assessment</div>'
'<div class="hero-subtitle">Advanced prediction powered by ensemble machine learning</div>'
'<div class="hero-stats">'
'<div class="hero-stat"><div class="hero-stat-value">0.90</div><div class="hero-stat-label">F1 Score</div></div>'
'<div class="hero-divider"></div>'
'<div class="hero-stat"><div class="hero-stat-value">0.95</div><div class="hero-stat-label">AUC-ROC</div></div>'
'<div class="hero-divider"></div>'
'<div class="hero-stat"><div class="hero-stat-value">1,700</div><div class="hero-stat-label">Records</div></div>'
'<div class="hero-divider"></div>'
'<div class="hero-stat"><div class="hero-stat-value">RF</div><div class="hero-stat-label">Model</div></div>'
'</div>'
'</div>',
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
