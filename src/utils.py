"""
Shared utility functions used throughout the project.

This module provides reusable helper functions for directory management,
file handling, JSON operations, and timestamp generation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def create_directory(path: str | Path) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str | Path
        Directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def file_exists(path: str | Path) -> bool:
    """
    Check whether a file or directory exists.

    Parameters
    ----------
    path : str | Path
        File or directory path.

    Returns
    -------
    bool
        True if the path exists, otherwise False.
    """
    return Path(path).exists()


def save_json(data: dict[str, Any], filepath: str | Path) -> None:
    """
    Save a dictionary as a JSON file.

    Parameters
    ----------
    data : dict
        Dictionary to save.
    filepath : str | Path
        Output JSON file path.
    """
    filepath = Path(filepath)
    create_directory(filepath.parent)

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str | Path) -> dict[str, Any]:
    """
    Load a JSON file.

    Parameters
    ----------
    filepath : str | Path
        Input JSON file path.

    Returns
    -------
    dict
        Loaded JSON data.
    """
    with Path(filepath).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, filepath: str | Path) -> None:
    """
    Save a pandas DataFrame as a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save.
    filepath : str | Path
        Output CSV file path.
    """
    filepath = Path(filepath)
    create_directory(filepath.parent)
    df.to_csv(filepath, index=False)


def timestamp() -> str:
    """
    Return the current timestamp.

    Returns
    -------
    str
        Timestamp in YYYY-MM-DD HH:MM:SS format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_section(title: str) -> None:
    """
    Print a formatted section heading.

    Parameters
    ----------
    title : str
        Section title.
    """
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")
