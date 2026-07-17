"""
Global configuration settings for the LLM Confidence Estimator project.

This module centralises project directory paths, model configuration and
global constants used throughout the experimental workflow.
"""

from pathlib import Path
import torch

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

OUTPUT_DIR: Path = PROJECT_ROOT / "Output"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
RESULTS_DIR: Path = PROJECT_ROOT / "result"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# ============================================================================
# Model Configuration
# ============================================================================

MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_BATCH_SIZE: int = 1

DEFAULT_MAX_NEW_TOKENS: int = 64

# ============================================================================
# Reproducibility
# ============================================================================

RANDOM_STATE: int = 42

CSV_ENCODING: str = "utf-8"

PLOT_DPI: int = 300

# ============================================================================
# Create Required Project Directories
# ============================================================================

for directory in (
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
