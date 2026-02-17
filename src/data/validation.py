"""Data validation functions for the raw liver disease DataFrame."""

import pandas as pd

from src.features.schema import (
    EXPECTED_COLUMNS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    TARGET,
)


class DataValidationError(Exception):
    """Raised when the DataFrame fails a validation check."""


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure the DataFrame has all required columns and no extra ones.

    Raises DataValidationError if columns are missing or unexpected.
    """
    expected = set(EXPECTED_COLUMNS)
    actual = set(df.columns)

    missing = expected - actual
    if missing:
        raise DataValidationError(f"Missing columns: {sorted(missing)}")

    extra = actual - expected
    if extra:
        raise DataValidationError(f"Unexpected extra columns: {sorted(extra)}")


def check_no_nulls(df: pd.DataFrame) -> None:
    """Raise DataValidationError if any column contains null values."""
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise DataValidationError(
            f"Null values found in: {cols_with_nulls.to_dict()}"
        )


def check_no_duplicates(df: pd.DataFrame) -> None:
    """Raise DataValidationError if duplicate rows are found."""
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        raise DataValidationError(f"Found {n_dupes} duplicate rows.")


def validate_target_distribution(df: pd.DataFrame) -> None:
    """Check target column contains only valid labels."""
    valid = {POSITIVE_LABEL, NEGATIVE_LABEL}
    actual = set(df[TARGET].unique())
    unexpected = actual - valid
    if unexpected:
        raise DataValidationError(
            f"Unexpected target values: {unexpected}. Expected: {valid}"
        )


def validate_numeric_ranges(df: pd.DataFrame) -> list[str]:
    """Return warnings (not errors) for numeric values outside expected ranges.

    Returns a list of warning strings; empty list means all ranges are fine.
    """
    expected_ranges: dict[str, tuple[float, float]] = {
        "Age":                (0,    120),
        "BMI":                (10,   60),
        "AlcoholConsumption": (0,    30),
        "PhysicalActivity":   (0,    20),
        "LiverFunctionTest":  (0,    150),
    }
    warnings = []
    for col, (lo, hi) in expected_ranges.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range:
            warnings.append(
                f"{col}: {out_of_range} values outside [{lo}, {hi}]"
            )
    return warnings


def run_all_validations(df: pd.DataFrame) -> list[str]:
    """Run all validation checks and return any warnings.

    Raises DataValidationError immediately for fatal issues (schema, nulls,
    target labels). Returns a list of non-fatal range warnings.
    """
    validate_schema(df)
    check_no_nulls(df)
    validate_target_distribution(df)
    return validate_numeric_ranges(df)
