
"""Utility functions to load and split the dataset for the Optimal Banking Model."""
from __future__ import annotations

import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

DEFAULT_TARGET = "target"

def load_data(csv_path: str, target_col: str = DEFAULT_TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a CSV and split into features X and target y.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    target_col : str
        Name of the target/label column (default: 'target').

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector as integers (0/1).
    """
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")
    y = (df[target_col] == 1).astype(int)
    X = df.drop(columns=[target_col]).copy()
    return X, y

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/validation split with sensible defaults."""
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)
