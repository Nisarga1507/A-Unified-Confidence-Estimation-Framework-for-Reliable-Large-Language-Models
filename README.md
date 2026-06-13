# Calibrated LLM Confidence Estimation

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Research](https://img.shields.io/badge/Research-MSc%20Dissertation-green)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# MSc Dissertation Project

## Confidence Estimation for Reliable Large Language Models

### Newcastle University — MSc Data Science

---

# Research Title

**Calibrated Multi-Signal Confidence Estimation for Hallucination Detection in Large Language Models**

---

# Overview

Large Language Models (LLMs) can generate fluent and coherent responses, but they frequently produce hallucinations — factually incorrect or unsupported information presented with high confidence.

This dissertation investigates how calibrated confidence estimation can improve the reliability and trustworthiness of language model outputs.

Rather than focusing solely on improving answer generation, this project focuses on estimating whether generated responses should be:

- accepted
- refined
- abstained from

based on calibrated confidence estimates.

The project is framed as a **selective prediction problem**, where the system learns when a model should trust its own outputs.

The repository is designed to support:

- reproducible experiments
- confidence calibration analysis
- hallucination evaluation
- reliability benchmarking
- modular experimentation

All experiments are designed to be fully reproducible using fixed dataset splits, logged predictions, and evaluation pipelines.

---

# Key Contribution

The primary contribution of this dissertation is a **calibrated multi-signal confidence estimation framework** that combines:

- internal uncertainty
- retrieval grounding
- verification consistency

into a unified confidence score for hallucination detection and reliability-aware decision-making in language models.

Unlike standard hallucination mitigation approaches that rely on isolated heuristics, this project systematically evaluates calibrated confidence aggregation strategies within a unified selective prediction framework.

---

# Research Problem

Current hallucination mitigation approaches such as Retrieval-Augmented Generation (RAG) and self-verification improve factual grounding but do not provide a reliable mechanism for determining whether generated responses are trustworthy.

Existing hallucination mitigation approaches such as Retrieval-Augmented Generation (RAG), SelfCheckGPT, and uncertainty estimation methods provide valuable reliability signals but are typically evaluated independently. Limited research has systematically investigated how multiple reliability signals can be calibrated and combined within a unified confidence estimation framework for selective prediction. This dissertation addresses this gap by studying confidence aggregation and calibration strategies for reliability-aware decision making in Large Language Models. or loosely combined heuristics.

While recent work explores confidence calibration and multi-signal hallucination detection, limited research systematically evaluates calibrated confidence aggregation strategies for selective prediction in language models.

This dissertation investigates whether multiple reliability signals can be integrated into a calibrated confidence estimation framework for trustworthy AI decision-making.

---

# Research Question

How can multiple reliability signals be integrated and calibrated to estimate the trustworthiness of language model outputs?

---

# Objectives

The project aims to:

1. Develop a reproducible baseline evaluation pipeline using TruthfulQA and an open-source instruction-tuned language model.

2. Investigate token-level uncertainty estimation as an initial confidence signal for hallucination detection.

3. Implement and evaluate additional reliability signals including retrieval agreement and verification consistency.

4. Design and compare confidence aggregation strategies for combining heterogeneous reliability signals.

5. Evaluate calibration quality and selective prediction performance using calibration and risk-aware metrics.
   

The objective is not only to improve hallucination detection performance, but also to ensure that predicted confidence scores align closely with empirical correctness probabilities.

---

# System Pipeline

```text
Question
   ↓
LLM Generation
   ↓
Baseline Prediction
   ↓
Reliability Signal Extraction
   ├── Token Uncertainty
   ├── Retrieval Agreement
   └── Verification Consistency
   ↓
Confidence Aggregation
   ↓
Calibration Layer
   ├── Temperature Scaling
   ├── Platt Scaling
   └── Isotonic Regression
   ↓
Selective Prediction
   ├── Accept
   ├── Refine
   └── Abstain
```

---

# Update Baseline Section

The baseline system evaluates Mistral-7B-Instruct on the TruthfulQA MC1 benchmark using constrained numerical option prediction. Token-level confidence scores are extracted during autoregressive generation and used for calibration analysis. Baseline experiments currently achieve approximately:

- Accuracy: 51.4%
- Expected Calibration Error (ECE): 0.2293

These results provide the reference point against which subsequent confidence estimation and calibration methods are evaluated.

# Reliability Signals

## 1. Token Uncertainty

Measures internal model uncertainty using mean token-level log probabilities extracted during autoregressive generation.

---

## 2. Retrieval Agreement

Measures semantic alignment between generated responses and retrieved supporting evidence using embedding-based similarity scoring.

---

## 3. Verification Consistency

Measures answer stability across multiple independently generated responses using a SelfCheckGPT-style verification strategy.

---

# Confidence Estimation

The framework evaluates both heuristic and learned confidence aggregation methods:

- Weighted linear aggregation
- Logistic regression confidence models
- Calibrated logistic regression
- Optional neural confidence combiners

Confidence estimators are trained using reliability signals extracted from token uncertainty, retrieval grounding, and verification consistency modules.
---

# Calibration

Confidence calibration ensures that predicted confidence scores align with empirical correctness probabilities.

The project evaluates calibration quality using:

- Expected Calibration Error (ECE)
- Reliability Diagrams
- Risk-Coverage Analysis
- Selective Prediction Metrics

Calibration methods evaluated include:

- Temperature Scaling
- Platt Scaling
- Isotonic Regression
  
---

# Decision Framework

The calibrated confidence score determines whether generated responses should be:

| Confidence Level | Action |
|---|---|
| High Confidence | Accept |
| Medium Confidence | Refine |
| Low Confidence | Abstain |

Thresholds are optimized using selective prediction and risk-coverage analysis.

---

# Datasets

## TruthfulQA

Primary benchmark for hallucination detection and truthfulness evaluation.

Repository:
https://github.com/sylinrl/TruthfulQA

---

## FEVER

Fact verification benchmark used for retrieval grounding and verification experiments.

Repository:
https://fever.ai/

---

# Evaluation Metrics

## Standard Metrics

- Accuracy
- Hallucination Rate
- Selective Accuracy

---

## Calibration Metrics

- Expected Calibration Error (ECE)
- Reliability Diagrams
- Brier Score

---

## Selective Prediction Metrics

- Coverage
- Selective Risk
- Risk-Coverage Tradeoff

---

# Repository Structure

```text
llm-confidence-estimator/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── notebooks/
|   ├── 01_baseline_truthfulqa.ipynb
│   ├── 02_option_likelihood_scoring.ipynb
│   ├── 03_calibration_analysis.ipynb
│   ├── 04_retrieval_agreement.ipynb
│   ├── 05_verification_consistency.ipynb
│   ├── 06_confidence_aggregation.ipynb
│   └── 07_selective_prediction.ipynb
|   
├── src/
│   ├── generation/
│   ├── retrieval/
│   ├── verification/
│   ├── confidence/
│   ├── calibration/
│   ├── evaluation/
│   └── utils/
│
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   ├── logs/
│   └── plots/
│
├── configs/
├── experiments/
├── figures/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Quick Start

## Clone Repository

```bash
git clone https://github.com/your-username/llm-confidence-estimator.git
cd llm-confidence-estimator
```

---

## Create Environment

```bash
python -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Baseline Evaluation

```bash
python src/generation/run_baseline.py
```

---

# Experimental Workflow

1. Baseline language model evaluation
2. Confidence logging pipeline
3. Calibration metric implementation
4. Reliability signal extraction
5. Confidence aggregation experiments
6. Threshold optimization
7. Reliability and calibration analysis

Each stage is evaluated independently before integration into the final confidence-aware framework.

---

# Current Progress

## Completed

- Repository setup
- Initial project architecture
- Baseline TruthfulQA notebook
- Literature review planning
- Evaluation pipeline design

---

## In Progress

- Baseline generation pipeline
- Token probability extraction
- ECE implementation
- Reliability diagram generation

---

## Planned

- Retrieval agreement implementation
- Verification consistency module
- Learned confidence combiners
- Calibration experiments
- Selective prediction evaluation

---

# Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- LangChain
- FAISS
- sentence-transformers
- scikit-learn
- matplotlib
- pandas
- NumPy

---

# Reproducibility

The repository is designed for reproducible experimentation.

Features include:

- fixed dataset splits
- deterministic experiment settings
- per-example prediction logging
- stored confidence outputs
- reusable evaluation scripts
- configuration-driven experiments

All generated outputs, predictions, and confidence estimates are logged for later recalibration and analysis.

---

# Expected Contribution

This dissertation contributes a unified calibration-aware confidence estimation framework that integrates token uncertainty, retrieval agreement, and verification consistency under shared evaluation conditions. The framework is designed to improve hallucination detection, confidence calibration, and selective prediction performance while supporting reliability-aware decision making in Large Language Models.

- model uncertainty
- retrieval grounding
- verification consistency

to improve hallucination detection and reliability-aware decision-making in Large Language Models.

The work shifts focus from answer generation toward calibrated answer reliability estimation.

---

# Future Work

Potential future extensions include:

- semantic entropy estimation
- adaptive retrieval strategies
- uncertainty-aware prompting
- larger benchmark evaluation
- multimodal reliability estimation
- confidence-aware AI agents

---

# Author

**Nisarga**  
MSc Data Science  
Newcastle University

---

# Supervisor Guidance

Research conducted under MSc dissertation supervision at Newcastle University.

---

# Citation

If referencing this repository, please cite appropriately after publication.

---

# License

This project is released under the MIT License.

# Implementation Status

| Component | Status |
|---|---|
| Baseline LLM Pipeline | In Progress |
| Token Uncertainty | In Progress |
| Retrieval Agreement | Planned |
| Verification Consistency | Planned |
| Calibration Metrics | In Progress |
| Reliability Diagrams | Planned |
