"""Optuna hyperparameter optimization for model selection."""

from collections.abc import Callable

import optuna
from sklearn.model_selection import cross_val_score

from src.models.train import build_pipeline

# ---------------------------------------------------------------------------
# Search-space suggestion helpers
# ---------------------------------------------------------------------------


def _suggest_random_forest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def _suggest_decision_tree(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }


def _suggest_svc(trial: optuna.Trial) -> dict:
    return {
        "C": trial.suggest_float("C", 0.01, 100.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
        "probability": True,
    }


def _suggest_logistic_regression(trial: optuna.Trial) -> dict:
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    params = {
        "C": trial.suggest_float("C", 0.001, 100.0, log=True),
        "penalty": penalty,
        "max_iter": 1000,
        "solver": "saga",
    }
    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    return params


SEARCH_SPACES: dict[str, Callable] = {
    "random_forest": _suggest_random_forest,
    "decision_tree": _suggest_decision_tree,
    "svc": _suggest_svc,
    "logistic_regression": _suggest_logistic_regression,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_objective(
    model_name: str,
    X_train,
    y_train,
    scoring: str = "f1_weighted",
    cv_folds: int = 5,
) -> Callable:
    """Return an Optuna objective function for the given model."""
    suggest_fn = SEARCH_SPACES[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        pipe = build_pipeline(model_name, params=params)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring=scoring)
        return scores.mean()

    return objective


def run_optimization(
    model_name: str,
    X_train,
    y_train,
    n_trials: int = 100,
    scoring: str = "f1_weighted",
    cv_folds: int = 5,
) -> optuna.Study:
    """Run Optuna optimization and return the study."""
    objective = create_objective(
        model_name, X_train, y_train, scoring=scoring, cv_folds=cv_folds
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study


def extract_best_params(study: optuna.Study) -> dict:
    """Extract the best hyperparameters from a completed study."""
    return dict(study.best_params)
