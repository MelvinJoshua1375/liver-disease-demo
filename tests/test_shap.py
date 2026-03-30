"""Tests for SHAP explanation module."""

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.evaluation.shap_explain import (
    aggregate_shap_importances,
    compute_shap_values,
    compute_single_shap,
    get_shap_explainer,
)


@pytest.fixture
def trained_pipeline(sample_data, preprocessor):
    """Create a fitted RF pipeline for SHAP tests."""
    X, y = sample_data
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X, y)
    return pipe


class TestGetShapExplainer:
    def test_returns_explainer_and_feature_names(self, trained_pipeline):
        explainer, names = get_shap_explainer(trained_pipeline)
        assert isinstance(explainer, shap.TreeExplainer)
        assert isinstance(names, list)
        assert len(names) == 10  # 5 numeric + 1 ordinal + 4 binary nominal

    def test_feature_names_match_preprocessor(self, trained_pipeline):
        _, names = get_shap_explainer(trained_pipeline)
        expected = list(trained_pipeline.named_steps["preprocessor"].get_feature_names_out())
        assert names == expected


class TestComputeShapValues:
    def test_returns_explanation(self, trained_pipeline, sample_data):
        X, _ = sample_data
        exp = compute_shap_values(trained_pipeline, X)
        assert isinstance(exp, shap.Explanation)

    def test_shape_matches_input(self, trained_pipeline, sample_data):
        X, _ = sample_data
        exp = compute_shap_values(trained_pipeline, X)
        assert exp.values.shape[0] == len(X)
        assert exp.values.shape[1] == 10  # transformed feature count

    def test_feature_names_present(self, trained_pipeline, sample_data):
        X, _ = sample_data
        exp = compute_shap_values(trained_pipeline, X)
        assert len(exp.feature_names) == 10

    def test_base_values_shape(self, trained_pipeline, sample_data):
        X, _ = sample_data
        exp = compute_shap_values(trained_pipeline, X)
        assert len(exp.base_values) == len(X)


class TestComputeSingleShap:
    def test_single_row(self, trained_pipeline, sample_data):
        X, _ = sample_data
        single = X.iloc[[0]]
        exp = compute_single_shap(trained_pipeline, single)
        assert exp.values.shape[0] == 1
        assert exp.values.shape[1] == 10

    def test_values_are_finite(self, trained_pipeline, sample_data):
        X, _ = sample_data
        exp = compute_single_shap(trained_pipeline, X.iloc[[0]])
        assert np.all(np.isfinite(exp.values))


class TestAggregateShapImportances:
    def test_returns_dataframe(self, trained_pipeline, sample_data):
        X, _ = sample_data
        exp = compute_shap_values(trained_pipeline, X)
        df = aggregate_shap_importances(exp)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "mean_abs_shap" in df.columns

    def test_sorted_descending(self, trained_pipeline, sample_data):
        X, _ = sample_data
        exp = compute_shap_values(trained_pipeline, X)
        df = aggregate_shap_importances(exp)
        assert df["mean_abs_shap"].is_monotonic_decreasing

    def test_all_features_present(self, trained_pipeline, sample_data):
        X, _ = sample_data
        exp = compute_shap_values(trained_pipeline, X)
        df = aggregate_shap_importances(exp)
        assert len(df) == 10

    def test_values_non_negative(self, trained_pipeline, sample_data):
        X, _ = sample_data
        exp = compute_shap_values(trained_pipeline, X)
        df = aggregate_shap_importances(exp)
        assert (df["mean_abs_shap"] >= 0).all()
