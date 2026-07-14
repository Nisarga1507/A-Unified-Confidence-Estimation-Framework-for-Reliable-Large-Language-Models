# Dataset Metadata

This directory documents the dataset characteristics and experimental configuration used throughout the dissertation.

## Dataset

| Property | Description |
|----------|-------------|
| Dataset | TruthfulQA Multiple-Choice Benchmark |
| Format | CSV |
| Language | English |
| Task | Multiple-choice Question Answering |
| Domain | Truthfulness Evaluation |

---

## Model Configuration

The experiments were conducted using:

- **Model:** Mistral-7B-Instruct
- **Framework:** Hugging Face Transformers
- **Programming Language:** Python 3.11

---

## Experimental Workflow

The dataset is used consistently throughout the complete experimental pipeline:

1. Baseline Confidence Estimation
2. Baseline Confidence Evaluation
3. Option Likelihood Signal Analysis
4. Multi-Signal Confidence Estimation
5. Final Comparative Evaluation

---

## Reproducibility

All preprocessing, inference, uncertainty estimation, statistical analysis and evaluation are implemented within the experimental notebooks. The metadata described here provides contextual information about the dataset and experimental setup used in the dissertation.
