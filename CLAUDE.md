# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Liver disease binary classification project using a synthetic Kaggle dataset (1700 records, 11 features).
Portfolio-ready architecture with TDD, Optuna HP tuning, WoE visualization, Streamlit Cloud deployment, and CI/CD.

**Dataset**: https://www.kaggle.com/datasets/rabieelkharoua/predict-liver-disease-1700-records-dataset

## Repository Structure

```
liver_disease_prediction-main/
├── src/                          # All reusable logic (single source of truth)
│   ├── config.py                 # load_settings() -> Settings dataclass from config/settings.yaml
│   ├── data/
│   │   ├── loader.py             # load_raw_data() -- maps int codes to string labels
│   │   ├── validation.py         # DataValidationError, run_all_validations()
│   │   └── splitter.py           # stratified_split(), extract_X_y()
│   ├── features/
│   │   ├── schema.py             # NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, etc.
│   │   ├── preprocessing.py      # create_preprocessor() -- canonical ColumnTransformer
│   │   ├── woe.py                # compute_woe_mappings(), apply_woe_mappings(), iv_summary_table()
│   │   └── woe_viz.py            # plot_woe_bars(), plot_iv_comparison(), validate_woe_monotonicity()
│   ├── models/
│   │   ├── train.py              # MODEL_REGISTRY, build_pipeline(), train_model()
│   │   ├── optimize.py           # Optuna: create_objective(), run_optimization(), extract_best_params()
│   │   ├── dt_lr_hybrid.py       # DTSegmentedLR (BaseEstimator + ClassifierMixin, no leakage)
│   │   └── persistence.py        # save_model(), load_model()
│   ├── evaluation/
│   │   ├── metrics.py            # compute_metrics(), compare_models(), plot_roc_curves()
│   │   ├── calibration.py        # plot_calibration_curves(), build_calibrated_pipeline()
│   │   └── cross_validation.py   # evaluate_with_cv(), compare_models_cv()
│   └── visualization/
│       ├── eda_plots.py          # crosstabulate(), plot_boxplot(), plot_correlation_matrix()
│       └── style.py              # PALETTE, MODEL_COLORS, matplotlib rcParams
├── app/
│   ├── app.py                    # Streamlit entry: 3 tabs (Prediction | Model Info | About)
│   └── components/
│       ├── prediction.py         # Two-column layout, risk gauge, feature chart
│       ├── model_info.py         # Metrics cards, comparison table, hyperparams
│       ├── about.py              # Dataset info, methodology, disclaimers
│       ├── styles.py             # Custom CSS injection
│       └── utils.py              # @st.cache_resource model loading, FEATURE_CONFIG, FEATURE_ORDER
├── notebooks/
│   ├── Exploratory_Data_Analysis.ipynb   # Lean: imports from src/, narrative + visuals only
│   └── Predictive_Modelling.ipynb        # Lean: model comparison, CV, Optuna, hybrid models
├── scripts/
│   ├── train_model.py            # End-to-end training -> models/liver_disease_model.pkl
│   ├── generate_metadata.py      # All models -> models/model_metadata.json
│   ├── generate_ppts.py          # Auto-generate EDA + Modelling PPTs via python-pptx
│   └── download_data.py          # Download dataset from Kaggle
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_data.py              # Data loading, validation, splitting (22 tests)
│   ├── test_features.py          # WoE, IV, preprocessing, monotonicity (22 tests)
│   ├── test_models.py            # Pipeline, training, DTSegmentedLR, Optuna (23 tests)
│   ├── test_evaluation.py        # Metrics, ROC, overfitting detection (9 tests)
│   └── test_app.py               # App imports, model loading, prediction output (14 tests)
├── models/
│   ├── liver_disease_model.pkl   # Trained RF pipeline (tracked in git for Streamlit Cloud)
│   └── model_metadata.json       # Metrics, comparison, feature importances
├── config/
│   └── settings.yaml             # Paths, feature lists, model params
├── .github/workflows/
│   ├── ci.yml                    # Lint (ruff) + test (pytest 80% cov) on push/PR
│   └── deploy.yml                # Artifact verification gate for Streamlit Cloud
├── .streamlit/config.toml        # Theme: primaryColor="#4A90D9", layout=wide
├── pyproject.toml                # Dependencies + pytest/ruff config
└── Makefile                      # make install|test|lint|train|run|generate-ppts
```

## Key Technical Details

**Dataset features:**
- Numeric: `Age`, `BMI`, `AlcoholConsumption`, `PhysicalActivity`, `LiverFunctionTest`
- Categorical: `Gender`, `Smoking`, `GeneticRisk` (ordinal: Low/Medium/High), `Diabetes`, `Hypertension`
- Target: `Diagnosis` (Positive/Negative -> 1/0)
- **Note**: CSV uses integer codes (0=Male, 1=Female, etc.) -- `load_raw_data()` maps them to strings

**Preprocessing** (canonical `create_preprocessor()`):
- `StandardScaler` for numeric features
- `OrdinalEncoder(categories=[["Low","Medium","High"]])` for `GeneticRisk`
- `OneHotEncoder(drop="if_binary")` for 4 nominal features -> 4 binary columns
- Total output: 10 features

**Best model:** Random Forest -- F1=0.897, AUC=0.953
- Hyperparameters: `max_depth=10, min_samples_split=10, min_samples_leaf=4, n_estimators=100`

**Critical bugs fixed (vs original notebooks):**
1. Data leakage in WoE: `compute_woe_mappings(train)` -> `apply_woe_mappings(test, mappings)`
2. Data leakage in DT+LR: `DTSegmentedLR` is a proper sklearn estimator -- fits inside pipeline
3. WoE smoothing: only applied when zero event/non-event counts exist (not always)
4. Preprocessor inconsistency: 3 different ColumnTransformers -> single `create_preprocessor()`
5. Overfitting threshold: hardcoded 0.1 -> statistical (gap > 2x test std dev across CV folds)

**DTSegmentedLR hybrid:**
- Shallow DT extracts leaf segments -> OneHotEncoded -> concatenated with features -> LR
- `BaseEstimator + ClassifierMixin` -- works in `cross_val_score` and `Pipeline`
- Inspired by Facebook GBDT+LR technique

**Optuna search spaces:** RF (n_estimators 50-500, max_depth 3-20), DT (max_depth 2-15),
SVC (C 0.01-100 log-uniform), LR (C 0.001-100, l1/l2/elasticnet)

## Commands

```bash
make install          # pip install -e .[dev]
make test             # pytest with coverage (80% minimum)
make lint             # ruff check src/ app/ scripts/ tests/
make train            # python scripts/train_model.py
make generate-ppts    # python scripts/generate_ppts.py
make run              # streamlit run app/app.py
```

## Architecture Notes

- **Single source of truth**: `src/features/schema.py` defines all feature lists. Never hardcode feature names elsewhere.
- **Preprocessor**: Always call `create_preprocessor()` from `src/features/preprocessing.py`. Never define a ColumnTransformer inline.
- **Splitting**: Only `stratified_split()` from `src/data/splitter.py`. Never call `train_test_split()` directly.
- **WoE pattern**: `compute_woe_mappings(train_df)` -> `apply_woe_mappings(df, mappings)`. Never fit on full dataset.
- **Feature order**: Always use `FEATURE_ORDER` from `app/components/utils.py` for Streamlit input DataFrame construction.
- **Model in repo**: `models/liver_disease_model.pkl` is tracked in git (~1-2MB) for Streamlit Cloud deployment.
- **Notebooks are lean**: They import everything from `src/`. No business logic lives in notebooks.
- **GitHub remote**: `https://github.com/MelvinJoshua1375/liver-disease-demo.git`
