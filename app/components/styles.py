"""Professional CSS design system for the Liver Disease Prediction app."""

import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Google Fonts ────────────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Keyframe animations ─────────────────────────────────────────── */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(16px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-20px); }
            to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(20px); }
            to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 0 0 rgba(29,78,216,0.3); }
            50%      { box-shadow: 0 0 0 8px rgba(29,78,216,0); }
        }
        @keyframes shimmer {
            0%   { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        @keyframes scaleIn {
            from { opacity: 0; transform: scale(0.92); }
            to   { opacity: 1; transform: scale(1); }
        }

        /* ── Base reset ───────────────────────────────────────────────────── */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        .stApp { background: #F1F5F9; }

        /* ── Hide Streamlit chrome ────────────────────────────────────────── */
        #MainMenu, footer, header, .stDeployButton { display: none !important; }

        /* ── Page container ───────────────────────────────────────────────── */
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 2rem !important;
            max-width: 1200px;
        }

        /* ══════════════════════════════════════════════════════════════════
           HERO BANNER
           ══════════════════════════════════════════════════════════════════ */
        .hero {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 50%, #1D4ED8 100%);
            padding: 2.5rem 3rem 2rem;
            margin: 0 -1rem 1.8rem -1rem;
            border-radius: 0 0 24px 24px;
            position: relative;
            overflow: hidden;
            animation: fadeIn 0.6s ease-out;
        }
        /* Decorative glow orb */
        .hero::before {
            content: '';
            position: absolute;
            top: -60px; right: -40px;
            width: 260px; height: 260px;
            background: radial-gradient(circle, rgba(59,130,246,0.25) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .hero::after {
            content: '';
            position: absolute;
            bottom: -30px; left: 20%;
            width: 180px; height: 180px;
            background: radial-gradient(circle, rgba(147,197,253,0.12) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.68rem;
            font-weight: 600;
            color: #93C5FD;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 14px;
            animation: fadeInUp 0.5s ease-out 0.15s both;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.5px;
            line-height: 1.15;
            margin: 0 0 8px 0;
            animation: fadeInUp 0.5s ease-out 0.25s both;
        }
        .hero-subtitle {
            font-size: 0.88rem;
            color: rgba(147,197,253,0.9);
            font-weight: 400;
            margin: 0;
            animation: fadeInUp 0.5s ease-out 0.35s both;
        }
        .hero-stats {
            display: flex;
            gap: 24px;
            margin-top: 18px;
            animation: fadeInUp 0.5s ease-out 0.45s both;
        }
        .hero-stat {
            text-align: center;
        }
        .hero-stat-value {
            font-size: 1.4rem;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.2;
        }
        .hero-stat-label {
            font-size: 0.65rem;
            font-weight: 500;
            color: rgba(147,197,253,0.7);
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        .hero-divider {
            width: 1px;
            background: rgba(255,255,255,0.15);
            align-self: stretch;
        }

        /* ══════════════════════════════════════════════════════════════════
           TAB BAR
           ══════════════════════════════════════════════════════════════════ */
        div[data-baseweb="tab-list"] {
            background: #FFFFFF;
            border-radius: 14px;
            padding: 5px;
            gap: 4px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
            animation: fadeInUp 0.4s ease-out 0.3s both;
        }
        div[data-baseweb="tab"] {
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.84rem !important;
            padding: 10px 24px !important;
            color: #64748B !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        div[data-baseweb="tab"]:hover {
            background: #F1F5F9 !important;
            color: #1E3A5F !important;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #1E3A5F, #1D4ED8) !important;
            color: #FFFFFF !important;
            box-shadow: 0 2px 8px rgba(29,78,216,0.3) !important;
        }
        div[data-baseweb="tab-highlight"],
        div[data-baseweb="tab-border"] { display: none !important; }

        /* ══════════════════════════════════════════════════════════════════
           CARDS
           ══════════════════════════════════════════════════════════════════ */
        .card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 24px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.03);
            margin-bottom: 1rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeInUp 0.4s ease-out both;
        }
        .card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.06), 0 8px 32px rgba(0,0,0,0.06);
            transform: translateY(-2px);
            border-color: #CBD5E1;
        }
        .card-title {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.3px;
            color: #94A3B8;
            margin-bottom: 16px;
        }

        /* ══════════════════════════════════════════════════════════════════
           SECTION HEADERS
           ══════════════════════════════════════════════════════════════════ */
        .section-header {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.4px;
            color: #94A3B8;
            margin: 24px 0 12px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .section-header::after {
            content: '';
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, #E2E8F0, transparent);
        }

        /* ══════════════════════════════════════════════════════════════════
           RISK RESULT BANNERS
           ══════════════════════════════════════════════════════════════════ */
        .risk-low, .risk-medium, .risk-high {
            border-radius: 16px;
            padding: 20px 24px;
            animation: scaleIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) both;
            transition: transform 0.2s ease;
        }
        .risk-low:hover, .risk-medium:hover, .risk-high:hover {
            transform: scale(1.01);
        }
        .risk-low {
            background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
            border: 1px solid #A7F3D0;
            border-left: 5px solid #059669;
            color: #064E3B;
        }
        .risk-medium {
            background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
            border: 1px solid #FDE68A;
            border-left: 5px solid #D97706;
            color: #78350F;
        }
        .risk-high {
            background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
            border: 1px solid #FECACA;
            border-left: 5px solid #DC2626;
            color: #7F1D1D;
        }
        .risk-label {
            font-size: 1.35rem;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .risk-desc {
            font-size: 0.88rem;
            font-weight: 500;
            opacity: 0.85;
        }

        /* ══════════════════════════════════════════════════════════════════
           PROBABILITY DISPLAY
           ══════════════════════════════════════════════════════════════════ */
        .prob-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 12px;
            animation: fadeInUp 0.4s ease-out 0.2s both;
        }
        .prob-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 16px 20px;
            text-align: center;
            transition: all 0.25s ease;
        }
        .prob-card:hover {
            border-color: #93C5FD;
            box-shadow: 0 2px 12px rgba(59,130,246,0.1);
        }
        .prob-card-label {
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #94A3B8;
            margin-bottom: 4px;
        }
        .prob-card-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #1E3A5F;
        }
        .prob-card-value.positive { color: #DC2626; }
        .prob-card-value.negative { color: #059669; }

        /* ══════════════════════════════════════════════════════════════════
           KPI METRIC CARDS
           ══════════════════════════════════════════════════════════════════ */
        div[data-testid="metric-container"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 16px 20px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        div[data-testid="metric-container"]:hover {
            border-color: #93C5FD;
            box-shadow: 0 4px 16px rgba(59,130,246,0.1);
            transform: translateY(-2px);
        }
        div[data-testid="metric-container"] label {
            font-size: 0.68rem !important;
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

        /* ══════════════════════════════════════════════════════════════════
           PRIMARY BUTTON
           ══════════════════════════════════════════════════════════════════ */
        div.stButton > button[kind="primary"] {
            width: 100%;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            padding: 0.8rem 1.2rem !important;
            border-radius: 12px !important;
            background: linear-gradient(135deg, #1E3A5F, #1D4ED8) !important;
            border: none !important;
            letter-spacing: 0.4px;
            box-shadow: 0 4px 14px rgba(29,78,216,0.35) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative;
            overflow: hidden;
        }
        div.stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 24px rgba(29,78,216,0.45) !important;
            transform: translateY(-2px);
        }
        div.stButton > button[kind="primary"]:active {
            transform: translateY(0) !important;
            box-shadow: 0 2px 8px rgba(29,78,216,0.3) !important;
        }

        /* ══════════════════════════════════════════════════════════════════
           INPUT FIELDS
           ══════════════════════════════════════════════════════════════════ */
        div[data-baseweb="input"] {
            border-radius: 10px !important;
            border-color: #E2E8F0 !important;
            transition: all 0.25s ease !important;
        }
        div[data-baseweb="input"]:focus-within {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
        }
        div[data-baseweb="select"] > div {
            border-radius: 10px !important;
            border-color: #E2E8F0 !important;
            transition: all 0.25s ease !important;
        }
        div[data-baseweb="select"] > div:focus-within {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
        }

        /* ── Slider styling ──────────────────────────────────────────────── */
        div[data-testid="stSlider"] > div > div > div[role="slider"] {
            background: linear-gradient(135deg, #1E3A5F, #1D4ED8) !important;
            border: 2px solid #FFFFFF !important;
            box-shadow: 0 2px 6px rgba(29,78,216,0.35) !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stSlider"] > div > div > div[role="slider"]:hover {
            transform: scale(1.2);
            box-shadow: 0 3px 10px rgba(29,78,216,0.45) !important;
        }
        div[data-testid="stSlider"] > div > div > div > div {
            background: linear-gradient(90deg, #1D4ED8, #3B82F6) !important;
        }

        /* ══════════════════════════════════════════════════════════════════
           EMPTY STATE PLACEHOLDER
           ══════════════════════════════════════════════════════════════════ */
        .empty-state {
            text-align: center;
            padding: 56px 32px;
            background: linear-gradient(135deg, #F8FAFC, #F1F5F9);
            border-radius: 20px;
            border: 2px dashed #CBD5E1;
            color: #94A3B8;
            animation: fadeIn 0.5s ease-out;
            transition: all 0.3s ease;
        }
        .empty-state:hover {
            border-color: #93C5FD;
            background: linear-gradient(135deg, #F8FAFC, #EFF6FF);
        }
        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 16px;
            animation: pulseGlow 2s ease-in-out infinite;
            display: inline-block;
        }
        .empty-state-title {
            font-weight: 700;
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 6px;
        }
        .empty-state-desc {
            font-size: 0.85rem;
            color: #94A3B8;
        }

        /* ══════════════════════════════════════════════════════════════════
           INFO / ABOUT CARDS
           ══════════════════════════════════════════════════════════════════ */
        .info-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 28px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 16px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeInUp 0.4s ease-out both;
        }
        .info-card:hover {
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            transform: translateY(-2px);
            border-color: #CBD5E1;
        }
        .info-card-icon {
            font-size: 1.6rem;
            margin-bottom: 12px;
        }
        .info-card-heading {
            font-size: 1rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 10px;
        }
        .info-card p, .info-card-body {
            font-size: 0.88rem;
            color: #64748B;
            line-height: 1.65;
        }

        /* ── Methodology timeline ────────────────────────────────────────── */
        .timeline-step {
            display: flex;
            gap: 14px;
            margin-bottom: 16px;
            padding: 12px 16px;
            border-radius: 12px;
            transition: all 0.25s ease;
        }
        .timeline-step:hover {
            background: #F8FAFC;
        }
        .timeline-num {
            flex-shrink: 0;
            width: 30px; height: 30px;
            background: linear-gradient(135deg, #1E3A5F, #1D4ED8);
            color: #FFFFFF;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .timeline-content {
            flex: 1;
        }
        .timeline-title {
            font-size: 0.88rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 2px;
        }
        .timeline-desc {
            font-size: 0.8rem;
            color: #64748B;
            line-height: 1.5;
        }

        /* ── Author card ─────────────────────────────────────────────────── */
        .author-card {
            background: linear-gradient(135deg, #0F172A, #1E3A5F);
            border-radius: 16px;
            padding: 24px;
            color: #FFFFFF;
            margin-top: 16px;
            animation: fadeInUp 0.4s ease-out 0.3s both;
        }
        .author-card:hover {
            box-shadow: 0 4px 20px rgba(15,23,42,0.3);
        }
        .author-name {
            font-size: 1.1rem;
            font-weight: 700;
        }
        .author-desc {
            font-size: 0.82rem;
            color: #93C5FD;
            margin-top: 4px;
        }

        /* ══════════════════════════════════════════════════════════════════
           EXPANDER, DATAFRAME, ALERT, DIVIDER, CAPTION
           ══════════════════════════════════════════════════════════════════ */
        details[data-testid="stExpander"] {
            border-radius: 12px !important;
            border-color: #E2E8F0 !important;
            transition: border-color 0.2s ease !important;
        }
        details[data-testid="stExpander"]:hover {
            border-color: #CBD5E1 !important;
        }
        hr { border-color: #E2E8F0 !important; }
        div[data-testid="stAlert"] { border-radius: 12px !important; }
        div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
        .stCaption { color: #94A3B8 !important; }

        /* ── Disclaimer banner ───────────────────────────────────────────── */
        .disclaimer {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 14px 20px;
            font-size: 0.82rem;
            color: #64748B;
            line-height: 1.55;
            margin-top: 16px;
        }

        /* ── Responsive ──────────────────────────────────────────────────── */
        @media (max-width: 768px) {
            .hero { padding: 2rem 1.5rem 1.5rem; }
            .hero-title { font-size: 1.6rem; }
            .hero-stats { gap: 16px; }
            .hero-stat-value { font-size: 1.1rem; }
            .prob-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
