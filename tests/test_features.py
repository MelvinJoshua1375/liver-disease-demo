"""Tests for WoE, IV, WoE monotonicity, and preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import create_preprocessor
from src.features.woe import (
    apply_woe_mappings,
    calculate_iv,
    calculate_woe,
    compute_woe_mappings,
    iv_summary_table,
)
from src.features.woe_viz import validate_woe_monotonicity

# ─── Helpers ─────────────────────────────────────────────────────────────────

def simple_df() -> pd.DataFrame:
    """A → 3 events / 1 non-event; B → 1 event / 5 non-events."""
    return pd.DataFrame({
        "feature": ["A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
        "target":  ["Pos", "Pos", "Pos", "Neg", "Pos", "Neg", "Neg", "Neg", "Neg", "Neg"],
    })


def zero_count_df() -> pd.DataFrame:
    """A → 3 events / 0 non-events; B → 0 events / 3 non-events."""
    return pd.DataFrame({
        "feature": ["A", "A", "A", "B", "B", "B"],
        "target":  ["Pos", "Pos", "Pos", "Neg", "Neg", "Neg"],
    })


def ordinal_df() -> pd.DataFrame:
    """Clear ordinal relationship: Low→mostly Neg, High→mostly Pos."""
    return pd.DataFrame({
        "GeneticRisk": (["Low"] * 100 + ["Medium"] * 100 + ["High"] * 100),
        "Diagnosis": (
            ["Negative"] * 80 + ["Positive"] * 20 +
            ["Negative"] * 50 + ["Positive"] * 50 +
            ["Negative"] * 20 + ["Positive"] * 80
        ),
    })


def sample_features_df(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
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
    })


# ─── TestWoECalculation ───────────────────────────────────────────────────────

class TestWoECalculation:
    def test_returns_dict_and_dataframe(self):
        mapping, detail = calculate_woe(simple_df(), "feature", "target", "Pos", "Neg")
        assert isinstance(mapping, dict)
        assert isinstance(detail, pd.DataFrame)
        assert "woe" in detail.columns

    def test_keys_match_categories(self):
        mapping, _ = calculate_woe(simple_df(), "feature", "target", "Pos", "Neg")
        assert set(mapping.keys()) == {"A", "B"}

    def test_manual_woe_no_smoothing_needed(self):
        """No zero counts → no smoothing. Verify against hand calculation.

        A: 3 events / 1 non-event (total 4 events, 6 non-events)
        dist_events_A  = 3/4 = 0.75
        dist_non_A     = 1/6 ≈ 0.1667
        WoE_A = ln(0.1667/0.75) ≈ -1.504

        B: 1 event / 5 non-events
        dist_events_B  = 1/4 = 0.25
        dist_non_B     = 5/6 ≈ 0.8333
        WoE_B = ln(0.8333/0.25) ≈ 1.204
        """
        mapping, _ = calculate_woe(simple_df(), "feature", "target", "Pos", "Neg")
        assert pytest.approx(mapping["A"], abs=0.01) == -1.504
        assert pytest.approx(mapping["B"], abs=0.01) == 1.204

    def test_smoothing_only_when_zero_counts(self):
        """When bins have zero counts, smoothing activates."""
        mapping, _ = calculate_woe(zero_count_df(), "feature", "target", "Pos", "Neg")
        # With smoothing=0.5:
        # A_events_adj=3.5, A_non_adj=0.5; B_events_adj=0.5, B_non_adj=3.5
        # dist_ev_A=3.5/4=0.875, dist_non_A=0.5/4=0.125 → WoE_A=ln(0.125/0.875)≈-1.946
        assert pytest.approx(mapping["A"], abs=0.01) == -1.946
        assert pytest.approx(mapping["B"], abs=0.01) == 1.946

    def test_no_smoothing_when_no_zeros(self):
        """Smoothing MUST NOT change values when no zero counts exist."""
        mapping_0, _ = calculate_woe(simple_df(), "feature", "target", "Pos", "Neg", smoothing=0.0)
        mapping_5, _ = calculate_woe(simple_df(), "feature", "target", "Pos", "Neg", smoothing=0.5)
        # No zeros → smoothing=0.0 and smoothing=0.5 must give identical results
        assert pytest.approx(mapping_0["A"], abs=1e-6) == mapping_5["A"]
        assert pytest.approx(mapping_0["B"], abs=1e-6) == mapping_5["B"]

    def test_numeric_feature_binning(self):
        """Numeric columns are auto-binned via qcut."""
        np.random.seed(42)
        df = pd.DataFrame({
            "score": np.random.randn(100),
            "target": np.random.choice(["Pos", "Neg"], 100),
        })
        mapping, _ = calculate_woe(df, "score", "target", "Pos", "Neg", bins=5)
        assert len(mapping) == 5
        assert all(isinstance(v, float) for v in mapping.values())


# ─── TestIVCalculation ────────────────────────────────────────────────────────

class TestIVCalculation:
    def test_returns_float(self):
        iv = calculate_iv(simple_df(), "feature", "target", "Pos", "Neg")
        assert isinstance(iv, float)

    def test_non_negative(self):
        iv = calculate_iv(simple_df(), "feature", "target", "Pos", "Neg")
        assert iv >= 0.0

    def test_sum_equals_components(self):
        df = pd.DataFrame({
            "f": ["A"] * 30 + ["B"] * 40 + ["C"] * 30,
            "t": ["Pos"] * 20 + ["Neg"] * 10 + ["Pos"] * 10 + ["Neg"] * 30 + ["Pos"] * 15 + ["Neg"] * 15,
        })
        iv = calculate_iv(df, "f", "t", "Pos", "Neg")
        _, detail = calculate_woe(df, "f", "t", "Pos", "Neg")
        assert pytest.approx(iv, abs=1e-9) == detail["iv_component"].sum()

    def test_iv_summary_sorted_descending(self):
        settings_df = pd.DataFrame({
            "A": ["X"] * 50 + ["Y"] * 50,
            "B": ["X"] * 90 + ["Y"] * 10,
            "target": ["Pos"] * 25 + ["Neg"] * 25 + ["Pos"] * 45 + ["Neg"] * 5,
        })
        table = iv_summary_table(settings_df, ["A", "B"], "target", positive_label="Pos", negative_label="Neg")
        assert table["Total_IV"].is_monotonic_decreasing


# ─── TestWoEReplacement ───────────────────────────────────────────────────────

class TestWoEReplacement:
    def test_compute_mappings_on_train_only(self):
        df = simple_df()
        mappings = compute_woe_mappings(df, ["feature"], "target", positive_label="Pos", negative_label="Neg")
        assert "feature" in mappings
        assert "A" in mappings["feature"]

    def test_apply_preserves_row_count(self):
        df = simple_df()
        mappings = compute_woe_mappings(df, ["feature"], "target", positive_label="Pos", negative_label="Neg")
        result = apply_woe_mappings(df, mappings)
        assert len(result) == len(df)

    def test_apply_changes_dtype_to_float(self):
        df = simple_df()
        mappings = compute_woe_mappings(df, ["feature"], "target", positive_label="Pos", negative_label="Neg")
        result = apply_woe_mappings(df, mappings)
        assert pd.api.types.is_float_dtype(result["feature"])

    def test_apply_handles_unseen_category(self):
        df_train = simple_df()
        mappings = compute_woe_mappings(df_train, ["feature"], "target", positive_label="Pos", negative_label="Neg")

        df_test = pd.DataFrame({
            "feature": ["A", "B", "C"],  # C is unseen
            "target": ["Pos", "Neg", "Pos"],
        })
        result = apply_woe_mappings(df_test, mappings, fill_value=0.0)
        assert result.loc[result.index[2], "feature"] == 0.0  # fill for "C"

    def test_train_test_separation_uses_train_mappings(self):
        df_train = simple_df()
        df_test = pd.DataFrame({
            "feature": ["A", "B", "B"],
            "target": ["Pos", "Neg", "Pos"],
        })
        mappings = compute_woe_mappings(df_train, ["feature"], "target", positive_label="Pos", negative_label="Neg")
        result_test = apply_woe_mappings(df_test, mappings)
        # Test uses train mappings, not recomputed
        assert pytest.approx(result_test["feature"].iloc[0]) == mappings["feature"]["A"]


# ─── TestPreprocessorConstruction ─────────────────────────────────────────────

class TestPreprocessorConstruction:
    def test_create_preprocessor_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        prep = create_preprocessor()
        assert isinstance(prep, ColumnTransformer)

    def test_preprocessor_fits_and_transforms(self):
        df = sample_features_df(80)
        y = np.random.choice([0, 1], 80)
        prep = create_preprocessor()
        X_train, X_test = df.iloc[:60], df.iloc[60:]
        prep.fit(X_train, y[:60])
        result = prep.transform(X_test)
        assert result.shape[0] == 20

    def test_preprocessor_output_shape(self):
        """5 numeric + 1 ordinal + 4 nominal (drop=if_binary → 1 each) = 10 columns."""
        df = sample_features_df(80)
        prep = create_preprocessor()
        prep.fit(df)
        result = prep.transform(df)
        assert result.shape[1] == 10

    def test_preprocessor_fit_on_train_only(self):
        """Fitting on train and transforming test should not raise errors."""
        df = sample_features_df(100)
        train, test = df.iloc[:80], df.iloc[80:]
        prep = create_preprocessor()
        prep.fit(train)
        result = prep.transform(test)
        assert result.shape[0] == 20


# ─── TestWoEMonotonicity ──────────────────────────────────────────────────────

class TestWoEMonotonicity:
    def test_ordinal_feature_is_monotonic_decreasing(self):
        """GeneticRisk with clear ordinal relationship → monotonically decreasing WoE."""
        result = validate_woe_monotonicity(
            ordinal_df(), "GeneticRisk", "Diagnosis",
            expected_direction="decreasing",
            categories=["Low", "Medium", "High"],
        )
        # With clear order Low→medium→High, and more Positive in High,
        # WoE should decrease (more events in High = lower WoE)
        assert result["is_monotonic"]
        assert result["direction"] == "decreasing"
        assert len(result["violations"]) == 0

    def test_returns_woe_values_list(self):
        result = validate_woe_monotonicity(simple_df(), "feature", "target", positive_label="Pos", negative_label="Neg")
        assert "woe_values" in result
        assert isinstance(result["woe_values"], list)
        assert len(result["woe_values"]) == 2

    def test_returns_is_monotonic_bool(self):
        result = validate_woe_monotonicity(simple_df(), "feature", "target", positive_label="Pos", negative_label="Neg")
        assert isinstance(result["is_monotonic"], bool)
