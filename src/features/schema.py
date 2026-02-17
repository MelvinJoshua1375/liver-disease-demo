"""Canonical feature definitions. Single source of truth for feature names and types."""

from typing import Final

NUMERIC_FEATURES: Final[list[str]] = [
    "Age",
    "BMI",
    "AlcoholConsumption",
    "PhysicalActivity",
    "LiverFunctionTest",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "Gender",
    "Smoking",
    "GeneticRisk",
    "Diabetes",
    "Hypertension",
]

ORDINAL_FEATURES: Final[dict[str, list[str]]] = {
    "GeneticRisk": ["Low", "Medium", "High"],
}

NOMINAL_FEATURES: Final[list[str]] = [
    "Gender",
    "Smoking",
    "Diabetes",
    "Hypertension",
]

TARGET: Final[str] = "Diagnosis"
POSITIVE_LABEL: Final[str] = "Positive"
NEGATIVE_LABEL: Final[str] = "Negative"

ALL_FEATURES: Final[list[str]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXPECTED_COLUMNS: Final[list[str]] = ALL_FEATURES + [TARGET]
