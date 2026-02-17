# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Liver disease binary classification project using a synthetic Kaggle dataset (1700 records, 11 features). The workflow spans EDA, statistical testing, model training/evaluation, and deployment via Streamlit.

## Repository Structure

- **Liver_Disease_EDA.ipynb** — Exploratory data analysis: descriptive stats, crosstabs, 100% stacked bar charts, t-tests/chi-square tests, correlation matrix, Weight of Evidence (WoE) transformation, and Information Value (IV) analysis
- **Liver_Disease_Modelling.ipynb** — Model training pipeline: sklearn preprocessing (StandardScaler, OneHotEncoder, OrdinalEncoder via ColumnTransformer), five classifiers (Logistic Regression, Decision Tree, SVC, Naive Bayes, Random Forest), ROC/AUC evaluation, model calibration (CalibratedClassifierCV for Decision Tree), overfitting detection, hyperparameter tuning, WoE-based Logistic Regression, Decision Tree segment features + Logistic Regression, and Streamlit app generation
- **requirements.txt** — Pinned dependencies for the Streamlit deployment app

## Key Technical Details

**Dataset features:**
- Numeric: Age, BMI, AlcoholConsumption, PhysicalActivity, LiverFunctionTest
- Categorical: Gender, Smoking, GeneticRisk (ordinal: Low/Medium/High), Diabetes, Hypertension
- Target: Diagnosis (Positive/Negative, mapped to 1/0)

**Best model:** Random Forest (F1=0.91, best AUC) with tuned hyperparameters: `max_depth=10, min_samples_split=10, min_samples_leaf=4, n_estimators=100`

**Deployed pipeline:** sklearn Pipeline with ColumnTransformer (StandardScaler for numerics, OneHotEncoder for categoricals) + RandomForestClassifier, serialized via joblib as `liver_disease_model.pkl`

**Notebooks were originally run on Google Colab** — data is loaded from `/content/drive/MyDrive/liver_disease_data.csv`. To run locally, update the CSV path.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit prediction app (requires liver_disease_model.pkl in working directory)
streamlit run app.py
```

## Architecture Notes

- Both notebooks define shared utility functions (`calculate_woe`, `calculate_iv`, `replace_with_woe`, `iv_summary_table`, `crosstabulate`) that are duplicated between them. Any changes to these functions should be applied in both notebooks.
- The Streamlit `app.py` is generated inline by the modelling notebook (cell 28) — it is not a standalone file in this repo. The app expects `liver_disease_model.pkl` in the same directory.
- Feature column ordering in the Streamlit app must match the pipeline's training order: `[Age, BMI, AlcoholConsumption, PhysicalActivity, LiverFunctionTest, Gender, Smoking, GeneticRisk, Diabetes, Hypertension]`.
