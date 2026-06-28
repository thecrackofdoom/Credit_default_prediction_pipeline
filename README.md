# Credit Default Risk Pipeline 🏦

End-to-end credit default risk pipeline on **Databricks Lakehouse**, featuring a Bronze/Silver/Gold medallion architecture orchestrated via **Spark Declarative Pipelines (SDP)**, **MLflow** experiment tracking, **XGBoost + Logistic Regression** models, and **SHAP**-based interpretability — benchmarked against the original Yeh & Lien (2009) academic baseline.

[![](https://img.shields.io/badge/Databricks-FF3621?logo=databricks&logoColor=white)]()
[![](https://img.shields.io/badge/PySpark-E25A1C?logo=apachespark&logoColor=white)]()
[![](https://img.shields.io/badge/Delta_Lake-003366?logo=delta&logoColor=white)]()
[![](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)]()
[![](https://img.shields.io/badge/XGBoost-EB5E28?logo=python&logoColor=white)]()
[![](https://img.shields.io/badge/SHAP-5C2D91?logo=python&logoColor=white)]()

---

## 📋 Table of Contents
- #-overview
- #-architecture
- #-key-results
- #-pipeline-walkthrough
- #-modeling-approach
- #-interpretability
- #-literature-benchmark
- #-repository-structure
- #-how-to-reproduce
- #-limitations--future-work

---

## 🎯 Overview

This project implements a production-style credit default prediction pipeline on Databricks, using the https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients (30,000 Taiwanese credit card clients, 23 features, October 2005 default outcome). The goal is to demonstrate **end-to-end data + ML engineering** — from raw ingestion to feature-engineered Gold tables, model registry, and interpretability — not just a one-off trained model.

**Why this dataset?** It is well-studied, allowing direct benchmarking against published baselines (Yeh & Lien, 2009) and identification of methodological pitfalls in later literature (see #-literature-benchmark).

---

## 🏗️ Architecture

docs/architecture.png

The pipeline follows the **Lakehouse medallion pattern**, with each layer materialized as a Delta table and orchestrated by **Spark Declarative Pipelines (SDP)**:

| Layer | Purpose | Format |
|-------|---------|--------|
| **Bronze** | Raw ingestion, schema preserved | Delta |
| **Silver** | Cleaned, categories consolidated, payment statuses normalized | Delta |
| **Gold** | One-hot encoded, ML-ready features | Delta |
| **MLflow** | Experiment tracking + Model Registry | — |


**SDP Pipeline Graph:**

![](docs/screenshots/sdp_pipeline.png)

---

## 📊 Key Results

### Discrimination Metrics

| Model | AUC | KS | Gini | Recall (Default) @ 0.5 |
|-------|------|------|------|------------------------|
| Logistic Regression (baseline) | 0.75 | 0.405 | 0.515 | 32% |
| **XGBoost (Normal Sampling) ⭐** | **0.78** | **0.44** | **0.57** | **59%** |
| XGBoost (Manual Oversampling) | 0.749 | 0.391 | 0.515 | 40% |
| XGBoost (SMOTE) | 0.759 | 0.393 | 0.519 | 38% |

> **Industry context:** KS > 0.30 is the typical retail credit scorecard validation threshold; KS = 0.43 is considered **strong**. Gini > 0.40 is standard for retail credit models.

![](docs/screenshots/ks_gini_metrics.png)

### Confusion Matrix + ROC (Best Model)

![](docs/screenshots/confusion_mat_roc_best.png)

### Sampling Strategy Tradeoff

![](docs/screenshots/scatter_accuracy_vs_recall_default_rate.png)

> Sampling-based class balancing improves recall on defaults at the cost of overall accuracy and probability calibration. The normal sampling strategy with `scale_pos_weight=3` was selected as the final model because it preserves probability calibration — critical when the score itself is the deliverable.

---

## 🔄 Pipeline Walkthrough

### 🥉 Bronze → Silver (`01_Bronze_ingest_Silver_clean.py`)
- Column renaming (`default.payment.next.month` → `de_pay`, `PAY_0` → `PAY_1`)
- **EDUCATION**: rare categories `{0, 5, 6}` consolidated into `4` (Others)
- **MARRIAGE**: unknown (`0`) consolidated into `3` (Others)
- **Payment status normalization**: `{-2, -1, 0}` → `0` (no delay); positive values = months of delay
- Data quality checks: null counts, record count parity vs Bronze, target distribution

### 🥈 Silver → Gold (`02_Silver_cleaned_Gold_features.py`)
- Drop non-predictive `ID` column
- One-hot encode `SEX`, `EDUCATION`, `MARRIAGE`
- Retain all numeric features (payment history, bill amounts, payment amounts)
- Output: feature-engineered, ML-ready Delta table

### 🤖 ML Training (`03_ML_training.py`)
- Stratified 70/30 train/test split (`random_state=100`)
- Logistic Regression baseline with `StandardScaler` + 5-fold CV
- XGBoost trained with **three class-imbalance strategies**:
  1. **Normal Sampling** — `scale_pos_weight=3`, regularized (`max_depth=4`, `gamma=0.2`)
  2. **Manual Oversampling** — minority class resampled to majority size
  3. **SMOTE** — synthetic minority oversampling
- Hyperparameters pre-tuned via `RandomizedSearchCV`
- All runs logged to MLflow with params, metrics, and model artifacts

---

## 🧠 Modeling Approach

### Why LR + XGBoost?
- **Logistic Regression**: interpretable baseline, regulator-friendly, sets the floor
- **XGBoost**: captures non-linearities and interactions; standard choice for tabular credit risk

### Class Imbalance Handling
The dataset has a **22% default rate** (imbalanced). I evaluated three strategies and selected `scale_pos_weight` over SMOTE because:
- SMOTE distorts probability calibration (synthetic neighbors don't reflect true joint distribution)
- For credit scoring, the **score is the product** — calibration matters more than raw recall

### Credit-Risk-Specific Metrics
Standard ML metrics (accuracy, F1) are misleading on imbalanced credit data. This project reports:
- **KS Statistic** — separation between default / non-default score distributions (banking standard)
- **Gini Coefficient** — `2 × AUC - 1`, used in regulatory submissions
- **AUC** — threshold-independent ranking quality

Implementations are in src/metrics.py.

---

## 🔍 Interpretability

### Global Feature Importance (XGBoost Native)
![](docs/screenshots/xgb_feature_importance.png)

### SHAP Summary (Global)
![](docs/screenshots/shap_feature_summary.png)

### Individual Prediction Explanations
**Waterfall plot** — cascading feature contributions for a single applicant:

![](docs/screenshots/shap_waterflow.png)

**Force plot** — additive decomposition from base value to prediction:

![](docs/screenshots/shap_force_plot.png)

> **Note:** SHAP values for XGBoost are in **log-odds space**, not probability. They show *direction* and *relative magnitude* of feature contribution to the default log-odds.

---

## 📚 Literature Benchmark

A core part of this project's due diligence was **benchmarking against published results** on the same dataset.

### Original Baseline — Yeh & Lien (2009)
The original paper that created this dataset reports a **lift-chart Area Ratio ≈ 0.54** for their best model (Neural Network), corresponding to **AUC ≈ 0.77**. The headline "R² = 0.9647" widely cited from this paper is a **calibration metric** (linear fit between predicted and smoothed actual probabilities), *not* classification accuracy — a subtle but important distinction.

This project's XGBoost AUC of **0.78** is at parity with the dataset's original baseline.

### Caution on Inflated Claims
Some later papers (e.g., Islam et al., 2018) report **>95% accuracy** on this dataset by engineering features from 6 months of payment history (`PAY_AMT1–6`, `BILL_AMT1–6`) that strongly **leak target information** — the target is October 2005 default behavior, which is directly caused by the same payment patterns being encoded as features. Notably, AUC/KS are absent from those results. This project deliberately follows Yeh & Lien's evaluation methodology to avoid such leakage.

> **Takeaway:** On UCI Credit Card Default, the realistic AUC ceiling without leakage is approximately **0.78–0.82**. Claims significantly above this should be scrutinized for target-derived features.

---

## 📁 Repository Structure

```
credit-risk-pipeline/
├── README.md
├── LICENSE
├── requirements.txt
│
├── notebooks/
│   ├── 01_Bronze_to_Silver.py             # Draft (interactive notebook version)
│   ├── 02_Silver_to_Gold.py               # Draft (interactive notebook version)
│   ├── 01_Bronze_ingest_Silver_clean.py   # Production (SDP)
│   ├── 02_Silver_cleaned_Gold_features.py # Production (SDP)
│   └── 03_ML_training.py                  # LR + XGB + MLflow + SHAP
│
├── src/
│   └── metrics.py                         # KS, Gini implementations
│
└── docs/
    ├── architecture.png
    └── screenshots/
        ├── sdp_pipeline.png
        ├── ks_gini_metrics.png
        ├── confusion_mat_roc_best.png
        ├── scatter_accuracy_vs_recall_default_rate.png
        ├── xgb_feature_importance.png
        ├── shap_feature_summary.png
        ├── shap_force_plot.png
        └── shap_waterflow.png
```

The `notebooks/` folder contains **two versions** of the Bronze→Silver and Silver→Gold logic:
- **Draft notebooks** (`01_Bronze_to_Silver.py`, `02_Silver_to_Gold.py`) — interactive, with quality-check cells and `display()` calls, useful for development
- **SDP notebooks** (`01_Bronze_ingest_Silver_clean.py`, `02_Silver_cleaned_Gold_features.py`) — production form using `@dp.materialized_view` decorators, orchestrated as a Spark Declarative Pipeline

---

## ⚙️ How to Reproduce

### Prerequisites
- Databricks workspace (Free Edition or paid trial)
- **Databricks Runtime 14.3 LTS ML** (or newer)
- UCI Credit Card Default dataset uploaded as `end_to_end_credit_default.uci_credit_card`

### Steps
1. Clone this repo and import the notebooks into your Databricks workspace
2. Create the SDP pipeline using the two `01_*_ingest_*` and `02_*_features` notebooks
3. Run the pipeline → produces `uci_credit_card_silver` and `uci_credit_card_gold` Delta tables
4. Run `03_ML_training.py` → trains models, logs to MLflow, generates SHAP plots
5. (Optional) Inspect the MLflow experiment to compare runs across sampling strategies

### Local Dependencies
```
pip install -r requirements.txt
```

---

## 🚧 Limitations & Future Work

**Limitations**
- **No temporal split**: train/test is random. Production credit models should use time-based splits to validate on future cohorts.
- **No external bureau features**: real scorecards incorporate credit bureau data (utilization, recent inquiries, delinquency history) which dominate predictive power.
- **No macroeconomic features**: default rates are highly sensitive to macro shocks (unemployment, GDP).
- **Batch-only scoring**: real-time serving via Databricks Model Serving is not implemented.
- **No drift monitoring**: PSI tracking on scored output would be added in production.

**Future Work**
- [ ] Threshold tuning with explicit business-cost framing (FP cost vs FN cost)
- [ ] WoE / IV encoding to align with industry-standard scorecard methodology
- [ ] Calibration analysis (reliability diagram, Brier score, isotonic calibration)
- [ ] Hyperparameter search with Optuna + MLflow autologging
- [ ] PSI-based drift monitoring on scored Delta output
- [ ] Power BI dashboard connected to scored Delta table via ODBC

---

## 📖 References

1. Yeh, I-Cheng & Lien, Che-hui (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.
2. UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients



---

*Built as a portfolio project to demonstrate end-to-end data engineering, ML lifecycle management, and credit risk modeling on the Databricks Lakehouse platform.*
