"""About tab: dataset description, methodology, and disclaimers."""

import streamlit as st


def render_about_tab() -> None:
    """Render the About tab."""
    st.subheader("About This Project")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Dataset")
        st.markdown(
            """
            **Source**: [Kaggle — Liver Disease Prediction Dataset](
            https://www.kaggle.com/datasets/rabieelkharoua/predict-liver-disease-1700-records-dataset)

            | Property | Value |
            |---------|-------|
            | Records | 1,700 |
            | Features | 10 |
            | Target | Diagnosis (Binary) |
            | Type | Synthetic |
            | Class Balance | 55% Positive / 45% Negative |
            """
        )

        st.markdown("### Features")
        st.markdown(
            """
            | Feature | Type | Description |
            |---------|------|-------------|
            | Age | Numeric | Patient age (years) |
            | BMI | Numeric | Body Mass Index |
            | AlcoholConsumption | Numeric | Units/week |
            | PhysicalActivity | Numeric | Hours/week |
            | LiverFunctionTest | Numeric | Lab score |
            | Gender | Binary | Male / Female |
            | Smoking | Binary | Yes / No |
            | GeneticRisk | Ordinal | Low / Medium / High |
            | Diabetes | Binary | Yes / No |
            | Hypertension | Binary | Yes / No |
            """
        )

    with col2:
        st.markdown("### Methodology")
        st.markdown(
            """
            1. **EDA** — Crosstabulations, 100% stacked bar charts, t-tests, chi-square tests,
               correlation matrix, WoE/IV analysis
            2. **Preprocessing** — StandardScaler (numeric), OrdinalEncoder (GeneticRisk),
               OneHotEncoder with dummy drop (nominal)
            3. **Model Comparison** — LR, Decision Tree, SVM, Naive Bayes, Random Forest,
               DT+LR Hybrid — evaluated with 5-fold stratified CV
            4. **Hyperparameter Tuning** — Optuna Bayesian optimization (100 trials per model)
            5. **Calibration** — CalibratedClassifierCV (Platt scaling) for Decision Tree
            6. **Overfitting Detection** — Flagged when train-test gap > 2× test score std dev

            **Best Model**: Random Forest (F1=0.90, AUC=0.96)
            """
        )

        st.markdown("### Author")
        st.markdown(
            """
            **Melvin Joshua**
            Machine Learning & Data Science Portfolio Project — 2026
            """
        )

        st.warning(
            "**Disclaimer**: This tool is for educational and demonstration purposes only. "
            "It is **not** intended for clinical diagnosis or medical decision-making. "
            "The dataset is synthetic and does not represent real patient data. "
            "Always consult a qualified healthcare professional.",
            icon="⚕️",
        )
