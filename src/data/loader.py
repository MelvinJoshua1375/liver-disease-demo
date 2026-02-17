"""Load raw CSV and decode numeric labels into human-readable strings."""

from pathlib import Path

import pandas as pd

from src.config import Settings

# The Kaggle dataset encodes all categorical columns as integers.
# These maps decode them back to the string labels expected by our pipeline.
_COLUMN_MAPS: dict[str, dict[int, str]] = {
    "Gender":       {0: "Male", 1: "Female"},
    "Smoking":      {0: "No",   1: "Yes"},
    "GeneticRisk":  {0: "Low",  1: "Medium", 2: "High"},
    "Diabetes":     {0: "No",   1: "Yes"},
    "Hypertension": {0: "No",   1: "Yes"},
    "Diagnosis":    {0: "Negative", 1: "Positive"},
}


def load_raw_data(settings: Settings) -> pd.DataFrame:
    """Load the raw CSV and map integer codes to string labels.

    Parameters
    ----------
    settings : Settings
        Project settings containing data.raw_path.

    Returns
    -------
    pd.DataFrame
        DataFrame with string labels for all categorical and target columns.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at settings.data.raw_path.
    """
    path = Path(settings.data.raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run `python scripts/download_data.py` to download it."
        )

    df = pd.read_csv(path)

    for col, mapping in _COLUMN_MAPS.items():
        if col in df.columns and pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].map(mapping)

    return df
