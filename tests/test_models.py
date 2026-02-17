"""Tests for model training, optimization, hybrid estimator, and persistence."""

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.models.dt_lr_hybrid import DTSegmentedLR
from src.models.optimize import (
    SEARCH_SPACES,
    create_objective,
    extract_best_params,
    run_optimization,
)
from src.models.persistence import load_metadata, load_model, save_metadata, save_model
from src.models.train import MODEL_REGISTRY, build_pipeline, get_model, train_model

# ---------------------------------------------------------------------------
# TestModelRegistry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_registry_has_at_least_five_models(self):
        assert len(MODEL_REGISTRY) >= 5

    def test_get_model_valid_name(self):
        model = get_model("random_forest")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_model_invalid_name_raises(self):
        with pytest.raises((KeyError, ValueError)):
            get_model("nonexistent_model")


# ---------------------------------------------------------------------------
# TestPipelineConstruction
# ---------------------------------------------------------------------------


class TestPipelineConstruction:
    def test_build_pipeline_returns_pipeline_with_two_steps(self):
        pipe = build_pipeline("random_forest")
        assert isinstance(pipe, Pipeline)
        step_names = [name for name, _ in pipe.steps]
        assert "preprocessor" in step_names
        assert "classifier" in step_names

    def test_pipeline_fit_predict(self, sample_data):
        X, y = sample_data
        pipe = build_pipeline("logistic_regression")
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (len(X),)


# ---------------------------------------------------------------------------
# TestTraining
# ---------------------------------------------------------------------------


class TestTraining:
    def test_train_model_returns_fitted_pipeline(self, sample_data):
        X, y = sample_data
        pipe = train_model(X, y, "random_forest")
        assert isinstance(pipe, Pipeline)
        # Should be fitted — predict should work without error
        pipe.predict(X[:5])

    def test_predictions_are_binary(self, sample_data):
        X, y = sample_data
        pipe = train_model(X, y, "decision_tree")
        preds = pipe.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, sample_data):
        X, y = sample_data
        pipe = train_model(X, y, "random_forest")
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_deterministic_with_same_seed(self, sample_data):
        X, y = sample_data
        pipe1 = train_model(X, y, "random_forest", params={"random_state": 0})
        pipe2 = train_model(X, y, "random_forest", params={"random_state": 0})
        np.testing.assert_array_equal(pipe1.predict(X), pipe2.predict(X))


# ---------------------------------------------------------------------------
# TestDTSegmentedLR
# ---------------------------------------------------------------------------


class TestDTSegmentedLR:
    def test_fits_without_error(self, sample_data, preprocessor):
        X, y = sample_data
        X_t = preprocessor.fit_transform(X)
        est = DTSegmentedLR()
        est.fit(X_t, y)

    def test_predict_shape(self, sample_data, preprocessor):
        X, y = sample_data
        X_t = preprocessor.fit_transform(X)
        est = DTSegmentedLR()
        est.fit(X_t, y)
        preds = est.predict(X_t)
        assert preds.shape == (len(X),)

    def test_predict_proba_shape(self, sample_data, preprocessor):
        X, y = sample_data
        X_t = preprocessor.fit_transform(X)
        est = DTSegmentedLR()
        est.fit(X_t, y)
        proba = est.predict_proba(X_t)
        assert proba.shape == (len(X), 2)

    def test_works_in_pipeline(self, sample_data, preprocessor):
        X, y = sample_data
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DTSegmentedLR()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (len(X),)

    def test_works_with_cross_val_score(self, sample_data, preprocessor):
        X, y = sample_data
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DTSegmentedLR()),
        ])
        scores = cross_val_score(pipe, X, y, cv=3, scoring="accuracy")
        assert scores.shape == (3,)
        assert all(0.0 <= s <= 1.0 for s in scores)


# ---------------------------------------------------------------------------
# TestModelPersistence
# ---------------------------------------------------------------------------


class TestModelPersistence:
    def test_save_load_roundtrip(self, sample_data, tmp_path):
        X, y = sample_data
        pipe = train_model(X, y, "logistic_regression")
        path = tmp_path / "model.joblib"
        save_model(pipe, path)
        loaded = load_model(path)
        np.testing.assert_array_equal(pipe.predict(X), loaded.predict(X))

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "missing.joblib")

    def test_metadata_roundtrip(self, tmp_path):
        meta = {"model": "rf", "accuracy": 0.95}
        path = tmp_path / "meta.json"
        save_metadata(meta, path)
        loaded = load_metadata(path)
        assert loaded == meta


# ---------------------------------------------------------------------------
# TestOptunaTuning
# ---------------------------------------------------------------------------


class TestOptunaTuning:
    def test_create_objective_returns_callable(self, sample_data):
        X, y = sample_data
        obj = create_objective("random_forest", X, y)
        assert callable(obj)

    def test_run_optimization_returns_study(self, sample_data):
        X, y = sample_data
        study = run_optimization("random_forest", X, y, n_trials=5)
        assert hasattr(study, "best_params")
        best = extract_best_params(study)
        assert isinstance(best, dict)

    @pytest.mark.parametrize("model_name", [
        "random_forest", "decision_tree", "svc", "logistic_regression",
    ])
    def test_search_space_defined(self, model_name):
        assert model_name in SEARCH_SPACES
