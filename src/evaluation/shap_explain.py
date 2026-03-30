"""SHAP-based model explanations for tree-based pipelines."""

import numpy as np
import pandas as pd
import shap


def get_shap_explainer(pipeline):
    """Create a SHAP TreeExplainer from a fitted sklearn pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Must contain a 'preprocessor' step and a 'classifier' step
        where the classifier is tree-based (RF, GBT, DT, etc.).

    Returns
    -------
    tuple[shap.TreeExplainer, list[str]]
        The explainer and the list of transformed feature names.
    """
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    explainer = shap.TreeExplainer(classifier)
    return explainer, feature_names


def compute_shap_values(pipeline, X: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values for a dataset through the full pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with 'preprocessor' and 'classifier' steps.
    X : pd.DataFrame
        Raw input features (before preprocessing).

    Returns
    -------
    shap.Explanation
        SHAP values for the positive class (index 1).
    """
    explainer, feature_names = get_shap_explainer(pipeline)
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    sv = explainer(X_transformed)

    # For binary classification, TreeExplainer returns shape (n, features, 2).
    # Extract positive-class SHAP values.
    if sv.values.ndim == 3:
        explanation = shap.Explanation(
            values=sv.values[:, :, 1],
            base_values=sv.base_values[:, 1] if sv.base_values.ndim == 2 else sv.base_values,
            data=X_transformed,
            feature_names=feature_names,
        )
    else:
        explanation = shap.Explanation(
            values=sv.values,
            base_values=sv.base_values,
            data=X_transformed,
            feature_names=feature_names,
        )
    return explanation


def compute_single_shap(pipeline, input_df: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values for a single observation.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline.
    input_df : pd.DataFrame
        Single-row DataFrame with raw feature values.

    Returns
    -------
    shap.Explanation
        Single-row SHAP explanation for the positive class.
    """
    return compute_shap_values(pipeline, input_df)


def aggregate_shap_importances(shap_explanation: shap.Explanation) -> pd.DataFrame:
    """Compute mean |SHAP| importances across all samples.

    Returns
    -------
    pd.DataFrame
        Columns: ['feature', 'mean_abs_shap'], sorted descending.
    """
    mean_abs = np.abs(shap_explanation.values).mean(axis=0)
    feature_names = shap_explanation.feature_names

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df
