"""Custom CSS for the Streamlit app."""

import streamlit as st


def inject_css() -> None:
    """Inject global custom CSS."""
    st.markdown(
        """
        <style>
        /* Metric cards */
        div[data-testid="metric-container"] {
            background: #F8F9FA;
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            padding: 12px 16px;
        }
        /* Tab styling */
        div[data-baseweb="tab-list"] {
            gap: 4px;
        }
        div[data-baseweb="tab"] {
            font-weight: 600;
            border-radius: 6px 6px 0 0;
        }
        /* Risk banners */
        .risk-low {
            background: #E8F5E9;
            border-left: 4px solid #4CAF50;
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            margin: 8px 0;
        }
        .risk-medium {
            background: #FFF8E1;
            border-left: 4px solid #FFC107;
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            margin: 8px 0;
        }
        .risk-high {
            background: #FFEBEE;
            border-left: 4px solid #F44336;
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            margin: 8px 0;
        }
        /* Hide Streamlit footer */
        footer {visibility: hidden;}
        /* Primary button */
        div.stButton > button[kind="primary"] {
            width: 100%;
            font-size: 1.1rem;
            padding: 0.6rem 1rem;
            border-radius: 8px;
        }
        /* Section headers */
        .section-header {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6C757D;
            margin-bottom: 8px;
            margin-top: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
