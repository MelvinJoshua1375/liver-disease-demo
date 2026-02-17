"""Model registry, pipeline construction, and training."""

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.features.schema import NOMINAL_FEATURES, NUMERIC_FEATURES, ORDINAL_FEATURES

MODEL_REGISTRY: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "svc": SVC,
    "naive_bayes": GaussianNB,
    "random_forest": RandomForestClassifier,
}


def _create_preprocessor() -> ColumnTransformer:
    """Build the canonical preprocessing ColumnTransformer."""
    ordinal_cols = list(ORDINAL_FEATURES.keys())
    ordinal_categories = list(ORDINAL_FEATURES.values())

    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
        ("nom", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                              sparse_output=False), NOMINAL_FEATURES),
    ])


def get_model(model_name: str, params: dict | None = None) -> ClassifierMixin:
    """Instantiate a model from the registry by name."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[model_name]
    return cls(**(params or {}))


def build_pipeline(
    model_name: str,
    params: dict | None = None,
    calibrate: bool = False,
) -> Pipeline:
    """Build a sklearn Pipeline with preprocessor + classifier."""
    preprocessor = _create_preprocessor()
    model = get_model(model_name, params)

    steps = [
        ("preprocessor", preprocessor),
        ("classifier", model),
    ]

    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV
        # Wrap the classifier with calibration
        steps[-1] = (
            "classifier",
            CalibratedClassifierCV(model, cv=3, method="sigmoid"),
        )

    return Pipeline(steps)


def train_model(
    X_train,
    y_train,
    model_name: str,
    params: dict | None = None,
    calibrate: bool = False,
) -> Pipeline:
    """Build a pipeline, fit it, and return the fitted pipeline."""
    pipe = build_pipeline(model_name, params=params, calibrate=calibrate)
    pipe.fit(X_train, y_train)
    return pipe
