"""Stratified train/test splitting. Single canonical splitting function."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.schema import TARGET


def stratified_split(
    df: pd.DataFrame,
    target_col: str = TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train and test with stratification on target.

    This is the ONLY place splitting happens in the project.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with target column present.
    target_col : str
        Name of the target column.
    test_size : float
        Fraction of data for the test set.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    train : pd.DataFrame
    test : pd.DataFrame
    """
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def extract_X_y(
    df: pd.DataFrame,
    target_col: str = TARGET,
    positive_label: str = "Positive",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and encode target as 0/1 integers.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    positive_label : str
        Label for the positive class (mapped to 1).

    Returns
    -------
    X : pd.DataFrame
        All columns except target.
    y : pd.Series
        Integer-encoded target (1 = positive, 0 = negative).
    """
    X = df.drop(columns=[target_col])
    y = (df[target_col] == positive_label).astype(int)
    return X, y
