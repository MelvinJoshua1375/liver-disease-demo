"""Canonical preprocessing pipeline. Single source of truth for feature transformations."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features.schema import (
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
)


def create_preprocessor() -> ColumnTransformer:
    """Build the canonical ColumnTransformer for the liver disease dataset.

    Transformations:
    - Numeric features    → StandardScaler
    - GeneticRisk         → OrdinalEncoder (Low=0, Medium=1, High=2)
    - Nominal binary cols → OneHotEncoder with drop="if_binary" to avoid
                            the dummy variable trap (4 cols → 4 output cols)

    Returns
    -------
    ColumnTransformer
        Unfitted transformer. Must be fit on TRAINING data only.
    """
    ordinal_col = list(ORDINAL_FEATURES.keys())[0]       # "GeneticRisk"
    ordinal_cats = list(ORDINAL_FEATURES.values())       # [["Low", "Medium", "High"]]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("scaler", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "ord",
                Pipeline([
                    ("ordinal", OrdinalEncoder(categories=ordinal_cats)),
                ]),
                [ordinal_col],
            ),
            (
                "nom",
                Pipeline([
                    ("onehot", OneHotEncoder(
                        drop="if_binary",
                        handle_unknown="ignore",
                        sparse_output=False,
                    )),
                ]),
                NOMINAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
