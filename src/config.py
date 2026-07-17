"""
Global configuration settings for the LLM Confidence Estimator project.

This module centralises commonly used paths, model configuration and
project-wide constants used throughout the experimental workflow.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "Output"

FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_DIR = PROJECT_ROOT / "result"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

RANDOM_STATE = 42
