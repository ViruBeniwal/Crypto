# Crypto Price Movement Classification

**Author:** Virendra Beniwal  
**Date:** July 2025  

## 1. Introduction

This project builds a machine learning pipeline to classify the short-term price movement of a cryptocurrency (e.g., BTC/USD) using historical minute-level data. Despite the noisy nature of crypto markets, the project demonstrates the full pipeline from raw data to evaluation.

## 2. Problem Statement

The task is to classify the future 5-minute return into one of 6 classes:

- **Class 0:** Large drop (< −0.2%)
- **Class 1:** Medium drop (−0.2% to −0.1%)
- **Class 2:** Small drop (−0.1% to 0%)
- **Class 3:** Small gain (0% to 0.1%)
- **Class 4:** Medium gain (0.1% to 0.2%)
- **Class 5:** Large gain (> 0.2%)

These thresholds were chosen arbitrarily for demonstration and can be adjusted to suit different trading strategies.

## 3. Data Preprocessing and Feature Engineering

- Raw data consisted of OHLCV (Open, High, Low, Close, Volume) values per minute.
- Future return over the next 5 minutes was calculated and used to assign class labels.
- Features included price slopes over rolling windows of 5, 10, and 30 minutes.

## 4. Modeling

A `RandomForestClassifier` from scikit-learn was used with the following hyperparameters:

- `n_estimators = 100`
- `max_depth = 8`
- `min_samples_leaf = 10`
- `class_weight = 'balanced'`
- `random_state = 42`

The model was trained on 80% of the data and evaluated on the remaining 20%.

## 5. Results

**Accuracy and LogLoss**

- **Accuracy:** 28.81%
- **LogLoss:** 1.6662


