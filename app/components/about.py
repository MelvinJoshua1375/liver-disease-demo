"""About tab: dataset description, methodology, and disclaimers."""

import streamlit as st


def render_about_tab() -> None:
    """Render the About tab with styled cards and timeline."""

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        # ── Dataset card ──────────────────────────────────────────────────
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-icon">📊</div>
                <div class="info-card-heading">Dataset</div>
                <div class="info-card-body">
                    <a href="https://www.kaggle.com/datasets/rabieelkharoua/predict-liver-disease-1700-records-dataset"
                       target="_blank" style="color:#1D4ED8; text-decoration:none; font-weight:600;">
                        Kaggle &mdash; Liver Disease Prediction Dataset &nearr;
                    </a>
                </div>
                <table style="width:100%; margin-top:12px; font-size:0.85rem; color:#475569; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #F1F5F9">
                        <td style="padding:6px 0; font-weight:600; color:#94A3B8">Records</td>
                        <td style="padding:6px 0; text-align:right">1,700</td>
                    </tr>
                    <tr style="border-bottom:1px solid #F1F5F9">
                        <td style="padding:6px 0; font-weight:600; color:#94A3B8">Features</td>
                        <td style="padding:6px 0; text-align:right">10</td>
                    </tr>
                    <tr style="border-bottom:1px solid #F1F5F9">
                        <td style="padding:6px 0; font-weight:600; color:#94A3B8">Target</td>
                        <td style="padding:6px 0; text-align:right">Diagnosis (Binary)</td>
                    </tr>
                    <tr style="border-bottom:1px solid #F1F5F9">
                        <td style="padding:6px 0; font-weight:600; color:#94A3B8">Type</td>
                        <td style="padding:6px 0; text-align:right">Synthetic</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 0; font-weight:600; color:#94A3B8">Class Balance</td>
                        <td style="padding:6px 0; text-align:right">55% Positive / 45% Negative</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Feature overview card ─────────────────────────────────────────
        st.markdown(
            """
            <div class="info-card" style="animation-delay: 0.1s">
                <div class="info-card-icon">🧬</div>
                <div class="info-card-heading">Features</div>
                <table style="width:100%; font-size:0.82rem; color:#475569; border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:2px solid #E2E8F0">
                            <th style="padding:6px 0; text-align:left; font-weight:700; color:#94A3B8; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px">Feature</th>
                            <th style="padding:6px 0; text-align:left; font-weight:700; color:#94A3B8; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px">Type</th>
                            <th style="padding:6px 0; text-align:left; font-weight:700; color:#94A3B8; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px">Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid #F1F5F9"><td style="padding:5px 0; font-weight:600">Age</td><td>Numeric</td><td>Patient age (years)</td></tr>
                        <tr style="border-bottom:1px solid #F1F5F9"><td style="padding:5px 0; font-weight:600">BMI</td><td>Numeric</td><td>Body Mass Index</td></tr>
                        <tr style="border-bottom:1px solid #F1F5F9"><td style="padding:5px 0; font-weight:600">AlcoholConsumption</td><td>Numeric</td><td>Units per week</td></tr>
                        <tr style="border-bottom:1px solid #F1F5F9"><td style="padding:5px 0; font-weight:600">PhysicalActivity</td><td>Numeric</td><td>Hours per week</td></tr>
                        <tr style="border-bottom:1px solid #F1F5F9"><td style="padding:5px 0; font-weight:600">LiverFunctionTest</td><td>Numeric</td><td>Lab score</td></tr>
                        <tr style="border-bottom:1px solid #F1F5F9"><td style="padding:5px 0; font-weight:600">Gender</td><td>Binary</td><td>Male / Female</td></tr>
                        <tr style="border-bottom:1px solid #F1F5F9"><td style="padding:5px 0; font-weight:600">Smoking</td><td>Binary</td><td>Yes / No</td></tr>
                        <tr style="border-bottom:1px solid #F1F5F9"><td style="padding:5px 0; font-weight:600">GeneticRisk</td><td>Ordinal</td><td>Low / Medium / High</td></tr>
                        <tr style="border-bottom:1px solid #F1F5F9"><td style="padding:5px 0; font-weight:600">Diabetes</td><td>Binary</td><td>Yes / No</td></tr>
                        <tr><td style="padding:5px 0; font-weight:600">Hypertension</td><td>Binary</td><td>Yes / No</td></tr>
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # ── Methodology timeline ──────────────────────────────────────────
        st.markdown(
            """
            <div class="info-card" style="animation-delay: 0.15s">
                <div class="info-card-icon">⚙️</div>
                <div class="info-card-heading">Methodology</div>

                <div class="timeline-step">
                    <div class="timeline-num">1</div>
                    <div class="timeline-content">
                        <div class="timeline-title">Exploratory Data Analysis</div>
                        <div class="timeline-desc">Crosstabulations, stacked bar charts, t-tests, chi-square tests, correlation matrix, WoE/IV analysis</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="timeline-num">2</div>
                    <div class="timeline-content">
                        <div class="timeline-title">Preprocessing</div>
                        <div class="timeline-desc">StandardScaler (numeric), OrdinalEncoder (GeneticRisk), OneHotEncoder with dummy drop (nominal)</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="timeline-num">3</div>
                    <div class="timeline-content">
                        <div class="timeline-title">Model Comparison</div>
                        <div class="timeline-desc">LR, Decision Tree, SVM, Naive Bayes, Random Forest, DT+LR Hybrid &mdash; 5-fold stratified CV</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="timeline-num">4</div>
                    <div class="timeline-content">
                        <div class="timeline-title">Hyperparameter Tuning</div>
                        <div class="timeline-desc">Optuna Bayesian optimization with 100 trials per model</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="timeline-num">5</div>
                    <div class="timeline-content">
                        <div class="timeline-title">Calibration</div>
                        <div class="timeline-desc">CalibratedClassifierCV (Platt scaling) for Decision Tree</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="timeline-num">6</div>
                    <div class="timeline-content">
                        <div class="timeline-title">Overfitting Detection</div>
                        <div class="timeline-desc">Flagged when train-test gap &gt; 2&times; test score standard deviation</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Author card ───────────────────────────────────────────────────
        st.markdown(
            """
            <div class="author-card">
                <div class="author-name">Melvin Joshua</div>
                <div class="author-desc">Machine Learning &amp; Data Science Portfolio Project &mdash; 2026</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Disclaimer ────────────────────────────────────────────────────
        st.markdown(
            """
            <div class="disclaimer" style="margin-top:16px">
                <strong>⚕️ Disclaimer:</strong> This tool is for educational and demonstration purposes only.
                It is <strong>not</strong> intended for clinical diagnosis or medical decision-making.
                The dataset is synthetic and does not represent real patient data.
                Always consult a qualified healthcare professional.
            </div>
            """,
            unsafe_allow_html=True,
        )
