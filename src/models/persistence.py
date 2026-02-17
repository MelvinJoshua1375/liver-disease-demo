"""Model and metadata persistence (save / load)."""

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_model(pipeline: Pipeline, path) -> None:
    """Save a fitted pipeline to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path) -> Pipeline:
    """Load a pipeline from disk. Raises FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def save_metadata(metadata_dict: dict, path) -> None:
    """Save metadata dictionary as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata_dict, f, indent=2)


def load_metadata(path) -> dict:
    """Load metadata dictionary from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with open(path) as f:
        return json.load(f)
