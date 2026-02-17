"""Tests for data loading, validation, and splitting."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_settings
from src.data.loader import _COLUMN_MAPS, load_raw_data
from src.data.splitter import extract_X_y, stratified_split
from src.data.validation import (
    DataValidationError,
    check_no_duplicates,
    check_no_nulls,
    run_all_validations,
    validate_numeric_ranges,
    validate_schema,
    validate_target_distribution,
)
from src.features.schema import EXPECTED_COLUMNS, POSITIVE_LABEL, TARGET

# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_valid_df(n: int = 50) -> pd.DataFrame:
    """Make a minimal valid DataFrame with string labels."""
    np.random.seed(0)
    return pd.DataFrame({
        "Age":                np.random.randint(20, 80, n),
        "BMI":                np.random.uniform(15, 40, n),
        "AlcoholConsumption": np.random.uniform(0, 20, n),
        "PhysicalActivity":   np.random.uniform(0, 10, n),
        "LiverFunctionTest":  np.random.uniform(20, 100, n),
        "Gender":     np.random.choice(["Male", "Female"], n),
        "Smoking":    np.random.choice(["Yes", "No"], n),
        "GeneticRisk":np.random.choice(["Low", "Medium", "High"], n),
        "Diabetes":   np.random.choice(["Yes", "No"], n),
        "Hypertension":np.random.choice(["Yes", "No"], n),
        "Diagnosis":  np.random.choice(["Positive", "Negative"], n),
    })


# ─── TestDataLoader ───────────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_maps_integer_codes_to_strings(self):
        """Loader must decode integer columns into string labels."""
        settings = load_settings()
        df = load_raw_data(settings)
        assert df["Gender"].dtype == object
        assert set(df["Gender"].unique()).issubset({"Male", "Female"})
        assert set(df["Smoking"].unique()).issubset({"Yes", "No"})
        assert set(df["GeneticRisk"].unique()).issubset({"Low", "Medium", "High"})
        assert set(df["Diagnosis"].unique()).issubset({"Positive", "Negative"})

    def test_load_returns_expected_columns(self):
        settings = load_settings()
        df = load_raw_data(settings)
        assert set(df.columns) == set(EXPECTED_COLUMNS)

    def test_load_missing_file_raises(self, tmp_path):
        settings = load_settings()
        settings.data.raw_path = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            load_raw_data(settings)

    def test_column_maps_cover_all_categoricals(self):
        """Ensure all expected categorical columns have a mapping defined."""
        from src.features.schema import CATEGORICAL_FEATURES
        for col in CATEGORICAL_FEATURES + [TARGET]:
            assert col in _COLUMN_MAPS, f"Missing mapping for {col}"


# ─── TestDataValidation ───────────────────────────────────────────────────────

class TestDataValidation:
    def test_validate_schema_passes_on_correct_df(self):
        validate_schema(make_valid_df())  # should not raise

    def test_validate_schema_missing_column(self):
        df = make_valid_df().drop(columns=["Age"])
        with pytest.raises(DataValidationError, match="Missing columns"):
            validate_schema(df)

    def test_validate_schema_extra_column(self):
        df = make_valid_df()
        df["extra_col"] = 0
        with pytest.raises(DataValidationError, match="Unexpected extra columns"):
            validate_schema(df)

    def test_check_no_nulls_clean(self):
        check_no_nulls(make_valid_df())  # should not raise

    def test_check_no_nulls_with_nulls(self):
        df = make_valid_df()
        df.loc[0, "Age"] = None
        with pytest.raises(DataValidationError, match="Null values"):
            check_no_nulls(df)

    def test_check_no_duplicates_clean(self):
        check_no_duplicates(make_valid_df())  # should not raise

    def test_check_no_duplicates_with_dupes(self):
        df = make_valid_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(DataValidationError, match="duplicate"):
            check_no_duplicates(df)

    def test_validate_target_valid_labels(self):
        validate_target_distribution(make_valid_df())  # should not raise

    def test_validate_target_unexpected_label(self):
        df = make_valid_df()
        df.loc[0, TARGET] = "Unknown"
        with pytest.raises(DataValidationError, match="Unexpected target values"):
            validate_target_distribution(df)

    def test_validate_numeric_ranges_returns_warnings_not_errors(self):
        df = make_valid_df()
        df.loc[0, "Age"] = 999  # out of range
        warnings = validate_numeric_ranges(df)
        assert any("Age" in w for w in warnings)

    def test_run_all_validations_clean_df(self):
        warnings = run_all_validations(make_valid_df())
        assert isinstance(warnings, list)


# ─── TestDataSplitter ─────────────────────────────────────────────────────────

class TestDataSplitter:
    def test_split_preserves_row_count(self):
        df = make_valid_df(100)
        train, test = stratified_split(df, test_size=0.2)
        assert len(train) + len(test) == len(df)

    def test_split_ratio_approximately_correct(self):
        df = make_valid_df(100)
        _, test = stratified_split(df, test_size=0.2)
        assert abs(len(test) - 20) <= 2

    def test_split_no_row_overlap(self):
        df = make_valid_df(100)
        train, test = stratified_split(df)
        # Use all columns to detect row overlap
        train_set = set(train.apply(tuple, axis=1))
        test_set = set(test.apply(tuple, axis=1))
        assert len(train_set & test_set) == 0

    def test_split_maintains_target_ratio(self):
        df = make_valid_df(200)
        train, test = stratified_split(df)
        full_ratio = (df[TARGET] == POSITIVE_LABEL).mean()
        train_ratio = (train[TARGET] == POSITIVE_LABEL).mean()
        test_ratio = (test[TARGET] == POSITIVE_LABEL).mean()
        assert abs(train_ratio - full_ratio) < 0.05
        assert abs(test_ratio - full_ratio) < 0.05

    def test_split_is_deterministic(self):
        df = make_valid_df(100)
        train1, test1 = stratified_split(df, random_state=42)
        train2, test2 = stratified_split(df, random_state=42)
        pd.testing.assert_frame_equal(train1, train2)

    def test_extract_X_y_shape(self):
        df = make_valid_df(50)
        X, y = extract_X_y(df)
        assert X.shape == (50, len(EXPECTED_COLUMNS) - 1)
        assert len(y) == 50

    def test_extract_X_y_target_is_binary(self):
        df = make_valid_df(50)
        _, y = extract_X_y(df)
        assert set(y.unique()).issubset({0, 1})
