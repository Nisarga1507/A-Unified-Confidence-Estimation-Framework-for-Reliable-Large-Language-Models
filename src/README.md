# Source Code

This directory contains the reusable Python modules supporting the experimental workflow presented in the dissertation.

The implementation has been modularised from the Jupyter notebooks to improve code organisation, readability and reproducibility.

Each module provides functionality used throughout one or more stages of the confidence estimation framework.

## Module Overview

| Module | Purpose |
|---------|---------|
| `config.py` | Stores project configuration, directory paths and global settings. |
| `utils.py` | Contains shared helper functions used across multiple notebooks. |
| `data_loader.py` | Loads and validates the TruthfulQA dataset. |
| `preprocessing.py` | Performs preprocessing and dataset preparation. |
| `model.py` | Loads and manages the Mistral-7B-Instruct model. |
| `confidence.py` | Implements baseline confidence estimation methods. |
| `uncertainty.py` | Computes uncertainty signals including entropy, margin score and probability gap. |
| `evaluation.py` | Implements evaluation metrics including ROC-AUC, AURC and selective prediction analysis. |
| `calibration.py` | Implements calibration analysis and Expected Calibration Error (ECE). |
| `statistics.py` | Performs statistical analysis and bootstrap evaluation. |
| `visualization.py` | Generates publication-quality figures used throughout the dissertation. |

The experimental notebooks utilise these reusable modules to reproduce the complete workflow presented in the dissertation while maintaining a clear separation between implementation, experimentation and analysis. This modular structure improves reproducibility, maintainability and code reuse across the project.
