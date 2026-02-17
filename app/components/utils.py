"""Shared utilities: model loading, feature config, feature ordering."""

import json
from pathlib import Path

import joblib
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH    = _PROJECT_ROOT / "models" / "liver_disease_model.pkl"
METADATA_PATH = _PROJECT_ROOT / "models" / "model_metadata.json"

FEATURE_CONFIG: dict = {
    "numeric": {
        "Age": {
            "min": 20, "max": 80, "default": 50, "step": 1, "type": int,
            "label": "Age (years)", "help": "Patient age",
        },
        "BMI": {
            "min": 15.0, "max": 45.0, "default": 27.7, "step": 0.1, "type": float,
            "label": "BMI (kg/m²)", "help": "Body Mass Index",
        },
        "AlcoholConsumption": {
            "min": 0.0, "max": 20.0, "default": 9.8, "step": 0.1, "type": float,
            "label": "Alcohol Consumption", "help": "Alcohol units per week",
        },
        "PhysicalActivity": {
            "min": 0.0, "max": 10.0, "default": 5.0, "step": 0.1, "type": float,
            "label": "Physical Activity (hrs/week)", "help": "Exercise hours per week",
        },
        "LiverFunctionTest": {
            "min": 20.0, "max": 100.0, "default": 59.9, "step": 0.1, "type": float,
            "label": "Liver Function Test Score", "help": "Liver function test result",
        },
    },
    "categorical": {
        "Gender":       {"options": ["Male", "Female"],        "default": "Male"},
        "Smoking":      {"options": ["No", "Yes"],             "default": "No"},
        "GeneticRisk":  {"options": ["Low", "Medium", "High"], "default": "Low"},
        "Diabetes":     {"options": ["No", "Yes"],             "default": "No"},
        "Hypertension": {"options": ["No", "Yes"],             "default": "No"},
    },
}

FEATURE_ORDER = [
    "Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "LiverFunctionTest",
    "Gender", "Smoking", "GeneticRisk", "Diabetes", "Hypertension",
]

_METADATA_FALLBACK = {
    "model_name": "Random Forest",
    "version": "1.0.0",
    "training_date": "2026-02-17",
    "dataset_size": 1700,
    "accuracy": 0.90,
    "f1_score": 0.90,
    "roc_auc": 0.96,
    "optimal_threshold": 0.50,
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
    },
}


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load and cache the trained pipeline. Returns None if file not found."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """Load model metadata with fallback to defaults."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return _METADATA_FALLBACK
