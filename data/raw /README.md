# Raw Dataset

This directory contains the original dataset used throughout this dissertation.

## Dataset

**TruthfulQA Multiple-Choice Benchmark**

- **File:** `TruthfulQA.csv`
- **Format:** CSV
- **Task:** Multiple-choice question answering
- **Purpose:** Benchmark dataset for evaluating confidence estimation in Large Language Models (LLMs)

## Usage

The raw **TruthfulQA Multiple-Choice** dataset serves as the input to the complete experimental workflow.

Specifically, it is first loaded in:

- **Notebook 1 – Baseline Confidence Estimation**

The predictions, confidence scores and uncertainty estimates generated from this dataset are subsequently analysed throughout the remaining notebooks:

- Notebook 2 – Baseline Confidence Evaluation
- Notebook 3 – Option Likelihood Signal Analysis
- Notebook 4 – Multi-Signal Confidence Estimation
- Notebook 5 – Final Comparative Evaluation

The original dataset remains unchanged throughout the project. All preprocessing and feature engineering are performed during notebook execution.
