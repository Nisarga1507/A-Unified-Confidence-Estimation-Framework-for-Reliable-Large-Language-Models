# LLM Confidence Estimator

## MSc Dissertation Project

**Confidence Estimation for Reliable Language Models**

---

## Project Overview

This project explores how to improve the reliability of large language models (LLMs) by designing a confidence estimation framework for detecting and reducing hallucinations.

LLMs are known to produce fluent responses that are sometimes factually incorrect. While techniques such as retrieval-augmented generation (RAG) and self-verification can improve performance, they do not provide a clear way to determine whether a given answer can be trusted.

The focus of this project is therefore on shifting from improving answer generation to evaluating answer reliability.

---

## Research Question

How can a confidence estimator be designed to determine whether an AI-generated answer should be accepted, refined, or rejected?

---

## Objectives

* Build a baseline language model on the TruthfulQA dataset
* Evaluate performance using accuracy and calibration metrics
* Extract confidence signals from model outputs (e.g., token probabilities)
* Develop a method for combining these signals into a single confidence score
* Compare different approaches for combining signals
* Design a simple decision framework (accept / refine / abstain)

---

## Dataset

* TruthfulQA (Multiple Choice format)
* A benchmark dataset designed to evaluate truthfulness in language models

---

## Model

* Initial baseline: Mistral-7B-Instruct (or LLaMA-3.1-8B-Instruct)
* The model should provide token-level probabilities for confidence estimation

---

## Evaluation Metrics

* Accuracy
* Confidence scores
* Expected Calibration Error (ECE)
* Reliability diagrams

These metrics are used to evaluate not only correctness, but also how well the model’s confidence reflects its actual performance.

---

## Pipeline

```
Question → Model → Prediction → Confidence → Evaluation
```

---

## Core Idea

The main idea is to build a confidence estimator based on multiple signals:

* Token probability (model uncertainty)
* Retrieval agreement (alignment with external knowledge) *(future work)*
* Verification consistency (stability across outputs) *(future work)*

These signals will be combined into a single confidence score that can be used to support decision-making.

---

## Methodology

### Phase 1: Baseline

* Run the model on TruthfulQA (MC format)
* Compute accuracy and basic confidence
* Implement ECE and reliability diagrams

### Phase 2: Signal Definition

* Define how token probability is computed
* Introduce additional signals if needed

### Phase 3: Combination Methods

* Weighted sum (baseline approach)
* Logistic regression
* Simple neural network (MLP)
* Calibration techniques (e.g., Platt scaling)

### Phase 4: Decision Framework

* Use confidence to decide: accept / refine / abstain
* Tune thresholds using validation data
* Evaluate trade-offs between accuracy and coverage

---

## Current Progress

* Repository created
* Project structure set up
* Initial plan defined

---

## Next Steps

* Implement baseline pipeline on TruthfulQA
* Run initial model predictions
* Extract token-level probabilities
* Compute calibration metrics (ECE)
* Visualise results using reliability diagrams

---

## Expected Contribution

* A framework for estimating the reliability of LLM outputs
* Comparison of different confidence estimation methods
* Insights into how confidence can be used for safer decision-making

---

## Project Structure

```
data/        → dataset references  
notebooks/   → experiments and baseline implementation  
src/         → core code  
outputs/     → results and plots  
```

---

## Summary

Overall, this project focuses on improving the trustworthiness of language models by enabling them to assess their own reliability, rather than only improving their ability to generate answers.
