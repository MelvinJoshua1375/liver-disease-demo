"""Load project configuration from YAML and resolve paths."""

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    raw_path: Path
    processed_dir: Path
    test_size: float
    random_state: int
    target_column: str
    positive_label: str
    negative_label: str


@dataclass
class ModelConfig:
    artifacts_dir: Path
    default_model: str
    random_state: int
    optuna_n_trials: int
    cv_folds: int


@dataclass
class Settings:
    data: DataConfig
    model: ModelConfig


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate settings from YAML config file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    data_cfg = raw["data"]
    model_cfg = raw["model"]

    return Settings(
        data=DataConfig(
            raw_path=PROJECT_ROOT / data_cfg["raw_path"],
            processed_dir=PROJECT_ROOT / data_cfg["processed_dir"],
            test_size=data_cfg["test_size"],
            random_state=data_cfg["random_state"],
            target_column=data_cfg["target_column"],
            positive_label=data_cfg["positive_label"],
            negative_label=data_cfg["negative_label"],
        ),
        model=ModelConfig(
            artifacts_dir=PROJECT_ROOT / model_cfg["artifacts_dir"],
            default_model=model_cfg["default_model"],
            random_state=model_cfg["random_state"],
            optuna_n_trials=model_cfg["optuna_n_trials"],
            cv_folds=model_cfg["cv_folds"],
        ),
    )
