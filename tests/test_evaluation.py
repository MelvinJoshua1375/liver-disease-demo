"""Tests for evaluation metrics, ROC data, and overfit detection."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.cross_validation import evaluate_with_cv
from src.evaluation.metrics import compute_metrics, compute_roc_data

# ---------------------------------------------------------------------------
# TestClassificationMetrics
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_returns_required_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = compute_metrics(y_true, y_pred)
        for key in ("accuracy", "f1_weighted", "f1_macro"):
            assert key in result

    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1, 1])
        result = compute_metrics(y, y)
        assert result["accuracy"] == 1.0

    def test_with_proba_includes_roc_auc(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])
        result = compute_metrics(y_true, y_pred, y_proba=y_proba)
        assert "roc_auc" in result

    def test_values_between_zero_and_one(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0])
        result = compute_metrics(y_true, y_pred)
        for v in result.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# TestROCData
# ---------------------------------------------------------------------------


class TestROCData:
    def test_returns_required_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.3, 0.7])
        result = compute_roc_data(y_true, y_proba)
        for key in ("fpr", "tpr", "thresholds", "auc", "optimal_threshold"):
            assert key in result

    def test_auc_between_zero_and_one(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_proba = np.array([0.2, 0.8, 0.3, 0.9, 0.7, 0.1])
        result = compute_roc_data(y_true, y_proba)
        assert 0.0 <= result["auc"] <= 1.0


# ---------------------------------------------------------------------------
# TestOverfitDetection
# ---------------------------------------------------------------------------


class TestOverfitDetection:
    def test_evaluate_with_cv_returns_required_keys(self, sample_data, preprocessor):
        X, y = sample_data
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
        result = evaluate_with_cv(pipe, X, y, cv_folds=3)
        assert "per_fold" in result
        assert "summary" in result
        assert "overfit_flags" in result

    def test_deep_tree_overfits(self, sample_data, preprocessor):
        X, y = sample_data
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth=None, random_state=42)),
        ])
        result = evaluate_with_cv(pipe, X, y, cv_folds=3)
        # An unrestricted tree typically overfits
        flags = result["overfit_flags"]
        assert any(flags.values()), (
            "Expected at least one overfit flag for an unrestricted tree"
        )

    def test_regularized_model_no_overfit(self, sample_data, preprocessor):
        X, y = sample_data
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(C=0.01, max_iter=1000)),
        ])
        scoring = {"accuracy": "accuracy"}
        result = evaluate_with_cv(pipe, X, y, cv_folds=5, scoring=scoring)
        flags = result["overfit_flags"]
        assert not flags["accuracy"], (
            "Expected no overfit flag on accuracy for a well-regularized model"
        )
