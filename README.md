# Customer Churn Prediction — Machine Learning Pipeline

An end-to-end machine learning project utilizing **scikit-learn** and **Logistic Regression** to predict customer churn in the telecommunications industry, evaluate business risk factors, and output actionable customer retention strategies.

Developed as an internship project submission for **CodeC Technologies**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Objective](#objective)
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Analysis](#exploratory-analysis)
- [Model Used](#model-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
  - [Performance Summary](#performance-summary)
  - [Confusion Matrix Breakdown](#confusion-matrix-breakdown)
  - [Visual Results](#visual-results)
- [Key Churn Factors](#key-churn-factors)
- [Project Structure](#project-structure)
- [How to Install](#how-to-install)
- [How to Run](#how-to-run)
  - [1. Train the Churn Model](#1-train-the-churn-model)
  - [2. Run Inference / Customer Scoring](#2-run-inference--customer-scoring)
- [Future Improvements](#future-improvements)

---

## Project Overview

Customer churn—the rate at which existing subscribers discontinue their service—poses a direct threat to recurring revenue in subscription-based telecom businesses. Acquiring new customers typically costs 5 to 7 times more than retaining existing ones. 

This project provides an automated, reproducible machine learning pipeline that identifies churn-prone customers, calculates calibrated churn probabilities, isolates top risk drivers, and prescribes targeted retention incentives before subscription lapse occurs.

---

## Business Problem

Telecommunications providers suffer significant annual revenue leakage when customers quietly cancel their subscriptions. Traditional reactive retention (contacting users after cancellation) fails because switching friction has already been overcome. 

To transition from **reactive** to **proactive retention**, businesses require:
1. **High Recall on Churners:** A model that identifies as many at-risk customers as possible (minimizing False Negatives).
2. **Explainability:** Business leaders must understand *why* a customer is likely to leave (e.g., month-to-month contract, lack of tech support, high monthly charges) to tailor retention offers effectively.
3. **Actionable Scoring:** An inference interface providing individual risk categories (`HIGH RISK`, `MODERATE RISK`, `LOW RISK`) and targeted retention recommendations.

---

## Objective

- Build, evaluate, and persist an end-to-end binary classification pipeline using scikit-learn.
- Preprocess heterogeneous demographic, account, and service features with standard scaling and one-hot encoding.
- Address class imbalance using balanced class weighting to maximize detection of churners (achieving **73.30% Recall** and **0.7607 ROC-AUC**).
- Quantify feature importance using Logistic Regression log-odds coefficients and odds ratios.
- Provide a command-line customer scoring tool (`predict.py`) that generates immediate retention strategies.

---

## Dataset Description

> **Dataset Source & Transparency Notice:**  
> This project uses a **reproducible synthetic telecom dataset** (`data.csv`, $N=7,043$ customer records) generated with a fixed random seed (`seed=42`).  
> The feature distributions and underlying log-odds churn mechanics are modeled closely after the public **IBM Telco Customer Churn dataset** to reflect real-world telecommunications consumer dynamics while guaranteeing **100% offline reproducibility**.

### Dataset Attributes

| Column Name | Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| `SeniorCitizen` | Numeric (Binary) | Whether the customer is a senior citizen | `0`, `1` |
| `Partner` | Categorical | Whether the customer has a partner | `Yes`, `No` |
| `Dependents` | Categorical | Whether customer has dependents | `Yes`, `No` |
| `tenure` | Numeric (Integer) | Months the customer has stayed with company | `0` to `72` months |
| `InternetService` | Categorical | Customer's internet service provider | `DSL`, `Fiber optic`, `No` |
| `TechSupport` | Categorical | Tech support add-on service | `Yes`, `No`, `No internet service` |
| `OnlineSecurity` | Categorical | Online security add-on service | `Yes`, `No`, `No internet service` |
| `PaperlessBilling`| Categorical | Whether paperless billing is enabled | `Yes`, `No` |
| `PaymentMethod` | Categorical | Customer's payment method | `Electronic check`, `Mailed check`, `Bank transfer`, `Credit card` |
| `Contract` | Categorical | Contract term duration | `Month-to-month`, `One year`, `Two year` |
| `MonthlyCharges` | Numeric (Float) | The amount charged to customer monthly | `$18.00` to `$130.00` |
| **`Churn`** | **Target (Binary)** | **Whether the customer churned** | **`0` (Retained), `1` (Churned)** |

- **Total Customers:** 7,043
- **Overall Churn Rate:** 27.13% (1,911 churned customers vs. 5,132 retained customers)
- **Train / Test Split:** Stratified 80% train (5,634 customers) / 20% test (1,409 customers)

---

## Data Preprocessing

Data preprocessing is encapsulated inside a reusable scikit-learn `ColumnTransformer` to prevent data leakage during cross-validation and testing:

1. **Categorical Feature Processing:**
   - Applied `OneHotEncoder(handle_unknown='ignore')` to 8 categorical columns (`Partner`, `Dependents`, `InternetService`, `TechSupport`, `OnlineSecurity`, `PaperlessBilling`, `PaymentMethod`, `Contract`).
   - Yields 22 binary dummy features capturing every service state.
2. **Numerical Feature Processing:**
   - Applied `StandardScaler()` to continuous numeric columns (`SeniorCitizen`, `tenure`, `MonthlyCharges`) to transform them to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$).
3. **Pipeline Encapsulation:**
   - The preprocessor and classifier are chained within a single `Pipeline` object and saved as `results/churn_model.joblib`.

---

## Exploratory Analysis

Key domain observations identified during dataset analysis:
- **Contract Duration:** Customers on **Month-to-month** contracts show dramatically higher churn rates (~42%) compared to customers on **Two-year** contracts (~7%).
- **Tenure:** Newer subscribers within the first 6–12 months exhibit the highest flight risk; churn likelihood drops exponentially as tenure surpasses 24 months.
- **Service Type:** Fiber optic subscribers experience higher monthly charges ($80–$110) and higher churn risk if they lack premium add-ons like Tech Support or Online Security.
- **Payment Method:** Customers using **Electronic check** churn at significantly higher rates than customers utilizing automated bank transfers or credit cards.

---

## Model Used

- **Algorithm:** **Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
- **Optimization Solver:** `lbfgs` with $L_2$ regularization
- **Max Iterations:** 1,000
- **Class Balancing:** `class_weight='balanced'`  
  *Rationale:* In subscription businesses, churn represents an imbalanced minority class (~27%). Standard models optimize for raw accuracy by ignoring churners. By applying inverse-frequency weighting, the loss function penalizes false negatives, ensuring high recall on actual churners.

---

## Evaluation Metrics

The model is evaluated using comprehensive classification metrics:
- **Accuracy:** Overall proportion of correct classifications.
- **Precision:** Of all customers predicted to churn, what fraction actually churned ($\frac{TP}{TP + FP}$).
- **Recall (Sensitivity):** Of all actual churners, what fraction was successfully identified ($\frac{TP}{TP + FN}$).
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve, measuring discrimination capability across all probability thresholds.
- **Confusion Matrix:** Counts of True Negatives, False Positives, False Negatives, and True Positives.

---

## Results

### Performance Summary

Evaluated on the held-out test split of **1,409 customers**:

| Metric | Score | Percentage | Business Interpretation |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | **0.7607** | **76.07%** | Strong ability to separate churners from loyal subscribers across all thresholds. |
| **Recall (Sensitivity)** | **0.7330** | **73.30%** | Successfully captures nearly **3 out of 4** at-risk churners. |
| **Accuracy** | **0.6799** | **67.99%** | Solid balanced accuracy across both classes under inverse weighting. |
| **Precision** | **0.4452** | **44.52%** | Balanced precision reflecting deliberate recall optimization for retention. |
| **F1-Score** | **0.5539** | — | Strong harmonic mean balancing recall and precision on imbalanced data. |

### Confusion Matrix Breakdown

| | Predicted Retained (`0`) | Predicted Churned (`1`) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Retained (`0`)** | **678** (True Negatives) | **349** (False Positives) | 1,027 |
| **Actual Churned (`1`)** | **102** (False Negatives) | **280** (True Positives) | 382 |
| **Total Predicted** | 780 | 629 | **1,409** |

- **280 out of 382 churners (73.30%)** were correctly intercepted before cancelling.

### Classification Report

```text
              precision    recall  f1-score   support

           0       0.87      0.66      0.75      1027
           1       0.45      0.73      0.55       382

    accuracy                           0.68      1409
   macro avg       0.66      0.70      0.65      1409
weighted avg       0.75      0.68      0.70      1409
```

### Visual Results

All evaluation plots are automatically generated and saved in `results/`:
- **`confusion_matrix.png`**: Heatmap illustrating true vs. predicted counts.
- **`roc_curve.png`**: ROC curve plotting True Positive Rate vs False Positive Rate ($AUC = 0.7607$).
- **`feature_importance.png`**: Horizontal bar plot highlighting top drivers increasing vs decreasing churn risk.

---

## Key Churn Factors

Feature importance was extracted directly from the Logistic Regression coefficients and translated into **Odds Ratios** ($OR = e^{\beta}$):

### 1. Factors Increasing Churn Risk (Top Risk Drivers)

| Feature | Coefficient ($\beta$) | Odds Ratio | Impact on Churn |
| :--- | :---: | :---: | :--- |
| **`Contract_Month-to-month`** | **+0.8521** | **2.34x** | Increases odds of churn by **134%** compared to baseline. |
| **`MonthlyCharges`** | **+0.4316** | **1.54x** | Each standard deviation increase in monthly charge raises churn odds by **54%**. |
| **`PaymentMethod_Electronic check`**| **+0.4086** | **1.50x** | Subscribers paying by manual electronic check have **50% higher** churn odds. |
| **`InternetService_Fiber optic`** | **+0.2200** | **1.25x** | Premium fiber users exhibit **25% higher** churn odds when pricing is high. |

### 2. Factors Decreasing Churn Risk (Top Retention Drivers)

| Feature | Coefficient ($\beta$) | Odds Ratio | Impact on Retention |
| :--- | :---: | :---: | :--- |
| **`Contract_Two year`** | **-0.9850** | **0.37x** | Reduces odds of churn by **63%**; strongest single retention factor. |
| **`tenure`** | **-0.4791** | **0.62x** | Each standard deviation increase in tenure reduces churn odds by **38%**. |
| **`TechSupport_Yes`** | **-0.4182** | **0.66x** | Having active technical support reduces churn odds by **34%**. |
| **`InternetService_No`** | **-0.2847** | **0.75x** | Basic voice-only subscribers have lower churn due to minimal service issues. |

---

## Project Structure

```text
customer-churn-prediction/
├── data.csv                  # Telecom customer dataset (7,043 rows, 12 columns)
├── churn_prediction.py       # Full preprocessing, training, evaluation, and plotting pipeline
├── predict.py                # Standalone customer scoring and risk classification script
├── requirements.txt          # Python package dependencies
├── .gitignore                # Git ignore configuration
├── README.md                 # Project documentation
└── results/                  # Generated model artifacts and evaluation plots
    ├── churn_model.joblib    # Serialized scikit-learn Pipeline
    ├── metrics.csv           # Evaluation metrics in CSV format
    ├── metrics.json          # Evaluation metrics in JSON format
    ├── classification_report.txt  # Full classification report
    ├── confusion_matrix.csv  # Confusion matrix values
    ├── confusion_matrix.png  # Heatmap of confusion matrix
    ├── roc_curve.png         # Receiver Operating Characteristic plot (AUC = 0.7607)
    ├── feature_importance.csv# All model coefficients and odds ratios
    └── feature_importance.png# Bar chart of key positive & negative churn drivers
```

---

## How to Install

1. **Clone the repository:**
   ```bash
   git clone https://github.com/charanhub888-star/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Train the Churn Model

Run the end-to-end training pipeline:

```bash
python churn_prediction.py
```

Outputs produced in `results/`:
- Serialized pipeline model (`churn_model.joblib`)
- Metrics summary (`metrics.csv`, `metrics.json`, `classification_report.txt`)
- Confusion matrix (`confusion_matrix.csv`, `confusion_matrix.png`)
- ROC curve plot (`roc_curve.png`)
- Feature importance ranking (`feature_importance.csv`, `feature_importance.png`)

### 2. Run Inference / Customer Scoring

Run scoring on pre-configured customer archetypes:

```bash
python predict.py
```

Score a specific customer profile from the command line:

```bash
python predict.py --contract "Month-to-month" --tenure 3 --monthly-charges 95.0 --internet "Fiber optic" --payment "Electronic check"
```

Example inference output:
```text
============================================================
PROFILE: Archetype 1 - New Month-to-Month Fiber Customer
============================================================
  SeniorCitizen       : 0
  Partner             : No
  Dependents          : No
  tenure              : 2
  InternetService     : Fiber optic
  TechSupport         : No
  OnlineSecurity      : No
  PaperlessBilling    : Yes
  PaymentMethod       : Electronic check
  Contract            : Month-to-month
  MonthlyCharges      : 89.5
------------------------------------------------------------
Predicted Outcome:       CHURN (1)
Churn Probability:       90.01%
Risk Tier:               HIGH RISK [CRITICAL ACTION REQUIRED]
Retention Strategy:      Offer contract migration discount (1-2 year agreement), complimentary premium tech support, and billing review.
============================================================
```

---

## Future Improvements

1. **Non-linear Ensemble Models:** Train and benchmark Random Forests, XGBoost, or LightGBM alongside Logistic Regression.
2. **Customer Lifetime Value (LTV) Weighting:** Factor individual subscriber revenue into churn risk scoring to prioritize high-value customer retention.
3. **Threshold Tuning:** Allow business managers to interactively adjust the classification probability threshold according to campaign budget constraints.
4. **Interactive Dashboard:** Build a Streamlit dashboard allowing customer service agents to score subscribers live during retention calls.
