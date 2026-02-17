"""Professional CSS design system for the Liver Disease Prediction app."""

import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Google Font ──────────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Base reset ───────────────────────────────────────────────── */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        .stApp { background-color: #F1F5F9; }

        /* ── Hide Streamlit chrome ────────────────────────────────────── */
        #MainMenu { visibility: hidden; }
        footer    { visibility: hidden; }
        header    { visibility: hidden; }
        .stDeployButton { display: none; }

        /* ── Page container ───────────────────────────────────────────── */
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 2rem !important;
            max-width: 1200px;
        }

        /* ── Hero banner ──────────────────────────────────────────────── */
        .hero {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 55%, #1D4ED8 100%);
            padding: 36px 48px 30px;
            margin: -4rem -4rem 2rem -4rem;
            border-radius: 0 0 28px 28px;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.5px;
            margin: 0 0 6px 0;
        }
        .hero-subtitle {
            font-size: 0.95rem;
            color: #93C5FD;
            font-weight: 400;
            margin: 0;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 0.72rem;
            font-weight: 600;
            color: #BAE6FD;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }

        /* ── Tab bar ──────────────────────────────────────────────────── */
        div[data-baseweb="tab-list"] {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 6px;
            gap: 4px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
            margin-bottom: 1.5rem;
        }
        div[data-baseweb="tab"] {
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 8px 20px !important;
            color: #64748B !important;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: #1E3A5F !important;
            color: #FFFFFF !important;
        }
        div[data-baseweb="tab-highlight"] { display: none !important; }
        div[data-baseweb="tab-border"]    { display: none !important; }

        /* ── Card ─────────────────────────────────────────────────────── */
        .card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 24px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
            margin-bottom: 1rem;
        }
        .card-title {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #94A3B8;
            margin-bottom: 16px;
        }

        /* ── Section divider label ────────────────────────────────────── */
        .section-header {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.4px;
            color: #94A3B8;
            margin: 20px 0 10px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .section-header::after {
            content: '';
            flex: 1;
            height: 1px;
            background: #E2E8F0;
        }

        /* ── Risk result banners ──────────────────────────────────────── */
        .risk-low {
            background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
            border: 1px solid #A7F3D0;
            border-left: 5px solid #059669;
            border-radius: 14px;
            padding: 20px 24px;
            color: #064E3B;
        }
        .risk-medium {
            background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
            border: 1px solid #FDE68A;
            border-left: 5px solid #D97706;
            border-radius: 14px;
            padding: 20px 24px;
            color: #78350F;
        }
        .risk-high {
            background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
            border: 1px solid #FECACA;
            border-left: 5px solid #DC2626;
            border-radius: 14px;
            padding: 20px 24px;
            color: #7F1D1D;
        }
        .risk-label {
            font-size: 1.4rem;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .risk-prob {
            font-size: 0.9rem;
            font-weight: 500;
            opacity: 0.85;
        }

        /* ── KPI metric cards ─────────────────────────────────────────── */
        div[data-testid="metric-container"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 16px 20px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        div[data-testid="metric-container"] label {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94A3B8 !important;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            color: #1E3A5F !important;
        }

        /* ── Primary button ───────────────────────────────────────────── */
        div.stButton > button[kind="primary"] {
            width: 100%;
            font-size: 1rem !important;
            font-weight: 700 !important;
            padding: 0.75rem 1rem !important;
            border-radius: 12px !important;
            background: linear-gradient(135deg, #1E3A5F, #1D4ED8) !important;
            border: none !important;
            letter-spacing: 0.3px;
            box-shadow: 0 4px 12px rgba(29,78,216,0.35) !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(29,78,216,0.45) !important;
            transform: translateY(-1px);
        }

        /* ── Input field styling ──────────────────────────────────────── */
        div[data-testid="stNumberInput"] > div,
        div[data-testid="stSelectbox"] > div > div {
            border-radius: 10px !important;
        }
        div[data-baseweb="input"] {
            border-radius: 10px !important;
            border-color: #CBD5E1 !important;
        }
        div[data-baseweb="input"]:focus-within {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
        }

        /* ── Progress bar (gauge) ─────────────────────────────────────── */
        div[data-testid="stProgress"] > div > div > div {
            border-radius: 100px;
        }
        div[data-testid="stProgress"] > div > div {
            border-radius: 100px;
            background: #E2E8F0;
        }

        /* ── Expander ─────────────────────────────────────────────────── */
        details[data-testid="stExpander"] {
            border-radius: 12px !important;
            border-color: #E2E8F0 !important;
        }

        /* ── Divider ──────────────────────────────────────────────────── */
        hr { border-color: #E2E8F0 !important; }

        /* ── Info/warning boxes ───────────────────────────────────────── */
        div[data-testid="stAlert"] {
            border-radius: 12px !important;
        }

        /* ── Dataframe ────────────────────────────────────────────────── */
        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        /* ── Caption ──────────────────────────────────────────────────── */
        .stCaption { color: #94A3B8 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
