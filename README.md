# Customer Feedback Analysis System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-TF--IDF_%26_Logistic_Regression-F7931E?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-API_Service-009688?style=for-the-badge)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge)

<p align="center">
  <b>An end-to-end Natural Language Processing (NLP) sentiment analysis pipeline designed to process raw customer feedback, perform TF-IDF vectorization, train class-balanced machine learning models, evaluate performance metrics, and deliver real-time predictions.</b>
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/Algorithm-TF--IDF_Vectorization-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-Logistic_Regression-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Handling-Class_Weight_Balanced-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Metrics-F1_Score_%26_Confusion_Matrix-red?style=for-the-badge" />
</p>

---

## Overview

Customer Feedback Analysis System is an NLP-driven sentiment classification pipeline built to analyze unstructured textual customer reviews and categorize them into actionable sentiment classes (Positive, Neutral, and Negative).

Instead of relying on simple keyword matching, the system builds an end-to-end data processing and model pipeline incorporating:

* Text Preprocessing and Normalization (Noise removal, tokenization, stop-word handling, negations)
* Feature Extraction (Sublinear TF-IDF N-gram Vectorization)
* Imbalance Handling (Automated inverse-frequency class weighting)
* Evaluation and Error Analysis (Precision, Recall, F1-Score, and Confusion Matrix benchmarking)

The system transforms raw feedback text into high-dimensional numerical feature vectors using TF-IDF Vectorization and predicts sentiment using Logistic Regression and LinearSVC models, accessible via modular CLI tools and a FastAPI backend service.

---

### Application Features

* Modular End-to-End Pipeline: Clean separation of data loading, preprocessing, feature extraction, model training, evaluation, and inference.
* Handled Class Imbalance: Incorporates class_weight="balanced" to address severe minority class scarcity across neutral and negative reviews.
* Transparent Evaluation Matrix: Outputs macro and weighted F1-scores, class-level precision and recall, and structured confusion matrices.
* Preserved Sentiment Signals: Optimized text cleaning pipelines that preserve critical sentiment negations (such as "not", "no", "never").
* Production-Ready Models: Serialized model artifacts (.pkl) ready for real-time inference via custom scripts or API endpoints.

---

## Project Objective

The primary objective is to build a scalable, production-ready sentiment analysis system that can:

* Clean and structure high-dimensional unstructured review text from raw CSV datasets.
* Address data quality issues like empty strings, null values, and uninformative stop-words.
* Transform textual review data into sparse numerical feature matrices using TfidfVectorizer.
* Mitigate real-world class imbalance where positive reviews heavily outnumber negative or neutral feedback.
* Evaluate performance using multi-class classification metrics (Precision, Recall, F1-Score, Confusion Matrix).
* Deploy an end-to-end Machine Learning pipeline structured clean for version control and modular reusability.

---

## Problem Statement

Analyzing customer feedback manually at scale is unfeasible, and simple keyword-matching approaches fail to capture context, leading to inaccurate insights for business decision-making.

Standard baseline models often fail to:

* Accurately detect minority classes (Negative and Neutral feedback) due to extreme class imbalance in user reviews.
* Handle text cleaning correctly without stripping away crucial sentiment modifiers like "not good" or "never buying again".
* Provide production-ready serialization pipelines for training, vectorization, and model inference.

### Proposed Solution

This project introduces a robust NLP classification pipeline that executes:

$$\text{Raw Reviews} \longrightarrow \text{Text Cleaning} \longrightarrow \text{TF-IDF Matrix} \longrightarrow \text{Balanced Model Training} \longrightarrow \text{Performance Evaluation}$$

For every review processed, the system produces:

```text
Cleaned Review Text
Predicted Sentiment Label (Positive / Neutral / Negative)
Class-Level Confidence / Decision Scores
Evaluation Benchmarks & Confusion Matrix Breakdown
