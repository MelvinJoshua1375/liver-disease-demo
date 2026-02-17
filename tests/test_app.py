"""Smoke tests for the Streamlit app: imports, model loading, prediction output."""

import numpy as np
import pandas as pd
import pytest

from app.components.utils import FEATURE_CONFIG, FEATURE_ORDER, load_metadata, load_model


class TestFeatureConfig:
    def test_feature_order_has_ten_features(self):
        assert len(FEATURE_ORDER) == 10

    def test_feature_order_matches_schema(self):
        from src.features.schema import ALL_FEATURES
        assert FEATURE_ORDER == ALL_FEATURES

    def test_numeric_config_has_required_keys(self):
        for feat, cfg in FEATURE_CONFIG["numeric"].items():
            for key in ("label", "min", "max", "default", "step", "type"):
                assert key in cfg, f"Missing key '{key}' for numeric feature '{feat}'"

    def test_categorical_config_has_required_keys(self):
        for feat, cfg in FEATURE_CONFIG["categorical"].items():
            for key in ("options", "default"):
                assert key in cfg, f"Missing key '{key}' for categorical feature '{feat}'"
            assert cfg["default"] in cfg["options"], f"Default not in options for '{feat}'"

    def test_numeric_defaults_within_range(self):
        for feat, cfg in FEATURE_CONFIG["numeric"].items():
            assert cfg["min"] <= cfg["default"] <= cfg["max"], (
                f"Default out of range for '{feat}': {cfg['default']}"
            )

    def test_genetic_risk_options_are_ordinal(self):
        options = FEATURE_CONFIG["categorical"]["GeneticRisk"]["options"]
        assert options == ["Low", "Medium", "High"]


class TestModelLoading:
    def test_load_model_returns_pipeline(self):
        model = load_model()
        assert model is not None

    def test_model_has_preprocessor_and_classifier(self):
        model = load_model()
        assert "preprocessor" in model.named_steps
        assert "classifier" in model.named_steps

    def test_model_predict_returns_binary(self):
        model = load_model()
        row = {feat: [FEATURE_CONFIG["numeric"][feat]["default"]]
               for feat in FEATURE_CONFIG["numeric"]}
        row.update({feat: [FEATURE_CONFIG["categorical"][feat]["default"]]
                    for feat in FEATURE_CONFIG["categorical"]})
        df = pd.DataFrame({feat: row[feat] for feat in FEATURE_ORDER})
        pred = model.predict(df)
        assert pred[0] in (0, 1)

    def test_model_predict_proba_sums_to_one(self):
        model = load_model()
        row = {feat: [FEATURE_CONFIG["numeric"][feat]["default"]]
               for feat in FEATURE_CONFIG["numeric"]}
        row.update({feat: [FEATURE_CONFIG["categorical"][feat]["default"]]
                    for feat in FEATURE_CONFIG["categorical"]})
        df = pd.DataFrame({feat: row[feat] for feat in FEATURE_ORDER})
        proba = model.predict_proba(df)[0]
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_model_proba_in_valid_range(self):
        model = load_model()
        rows = []
        for _ in range(5):
            row = {feat: [FEATURE_CONFIG["numeric"][feat]["default"]]
                   for feat in FEATURE_CONFIG["numeric"]}
            row.update({feat: [FEATURE_CONFIG["categorical"][feat]["default"]]
                        for feat in FEATURE_CONFIG["categorical"]})
            rows.append(pd.DataFrame({feat: row[feat] for feat in FEATURE_ORDER}))
        df = pd.concat(rows, ignore_index=True)
        proba = model.predict_proba(df)[:, 1]
        assert np.all((proba >= 0) & (proba <= 1))


class TestMetadataLoading:
    def test_load_metadata_returns_dict(self):
        meta = load_metadata()
        assert isinstance(meta, dict)

    def test_metadata_has_required_keys(self):
        meta = load_metadata()
        for key in ("model_name", "accuracy", "f1_score", "roc_auc"):
            assert key in meta, f"Missing metadata key: {key}"

    def test_metadata_metrics_in_valid_range(self):
        meta = load_metadata()
        for metric in ("accuracy", "f1_score", "roc_auc"):
            assert 0.0 <= meta[metric] <= 1.0, f"Metric '{metric}' out of [0,1]"

    def test_metadata_model_comparison_is_list(self):
        meta = load_metadata()
        if "model_comparison" in meta:
            assert isinstance(meta["model_comparison"], list)
            assert len(meta["model_comparison"]) > 0
