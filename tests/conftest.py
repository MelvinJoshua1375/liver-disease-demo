"""Shared test fixtures for model and evaluation tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


@pytest.fixture
def sample_data():
    """Generate reproducible sample data matching the project schema."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "Age": np.random.randint(20, 80, n),
        "BMI": np.random.uniform(15, 40, n),
        "AlcoholConsumption": np.random.uniform(0, 20, n),
        "PhysicalActivity": np.random.uniform(0, 10, n),
        "LiverFunctionTest": np.random.uniform(20, 100, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Smoking": np.random.choice(["Yes", "No"], n),
        "GeneticRisk": np.random.choice(["Low", "Medium", "High"], n),
        "Diabetes": np.random.choice(["Yes", "No"], n),
        "Hypertension": np.random.choice(["Yes", "No"], n),
    })
    y = np.random.choice([0, 1], n)
    return df, y


@pytest.fixture
def preprocessor():
    """Create a preprocessor matching the project schema."""
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(),
         ["Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "LiverFunctionTest"]),
        ("ord", OrdinalEncoder(categories=[["Low", "Medium", "High"]]),
         ["GeneticRisk"]),
        ("nom", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
         ["Gender", "Smoking", "Diabetes", "Hypertension"]),
    ])
