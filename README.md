# 🪙 Crypto Price Movement Classification

Author: **Virendra Beniwal**  
Date: **July 2025**

## 📌 Project Overview

This project aims to classify short-term price movement of a cryptocurrency (e.g., BTC/USD) using historical minute-level market data. The goal is to predict the **5-minute future return** and categorize it into one of six classes, representing different levels of price change.

Although predicting crypto movements is notoriously difficult due to market noise, this project demonstrates a complete machine learning pipeline — from feature extraction to model evaluation.

---

## 🧠 Problem Statement

We classify each 5-minute return into one of six discrete bins:

| Class | Description           | Return Range         |
|-------|------------------------|----------------------|
| 0     | Large drop             | < −0.2%              |
| 1     | Medium drop            | −0.2% to −0.1%       |
| 2     | Small drop             | −0.1% to 0%          |
| 3     | Small gain             | 0% to 0.1%           |
| 4     | Medium gain            | 0.1% to 0.2%         |
| 5     | Large gain             | > 0.2%               |

> **Note**: These bins are arbitrary and can be tuned for different trading strategies.

---

## 🧹 Data Preprocessing & Feature Engineering

- Used OHLCV (Open, High, Low, Close, Volume) minute-level data.
- Computed **5-minute future returns** to assign class labels.
- Engineered features:
  - Price slopes over rolling windows: 5, 10, and 30 minutes.
- Standardized all features using `StandardScaler`.
- Handled missing and infinite values.

---

## 🤖 Model

Used a `RandomForestClassifier` from **scikit-learn** with:

```python
n_estimators = 100
max_depth = 8
min_samples_leaf = 10
class_weight = 'balanced'
random_state = 42
