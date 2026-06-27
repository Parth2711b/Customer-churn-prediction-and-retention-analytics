# FintelliQ — Banking Intelligence Platform

FintelliQ is a data analytics platform built to address three high-cost problems in retail banking: customer churn, credit default, and transaction fraud. The first module, customer churn prediction, is complete. Credit default and fraud detection are in development.

The project covers the full analytics lifecycle from raw data through a deployed application, including data engineering, exploratory analysis, feature engineering, machine learning, explainability, experiment tracking, and a live Streamlit interface for bank operations teams.

## Repository Structure

```
FintelliQ/
├── app/
│   ├── app.py
│   ├── churn_model.pkl
│   ├── X_test_transformed.npy
│   └── y_test.csv
├── data/
│   ├── processed/
│   │   ├── churn_cleaned.csv
│   │   ├── churn_features.csv
│   │   ├── shap_bar.png
│   │   ├── shap_summary.png
│   │   ├── shap_waterfall.png
│   │   └── threshold_analysis.png
│   └── raw/
├── docs/
│   ├── Customer_Churn_Business_Case.docx
│   ├── Customer_Churn_Dashboard.html
│   ├── Customer_Churn_Project_Deck.html
│   └── FintelliQ_Dashboard.html
├── notebooks/
│   ├── EDA_churn.ipynb
│   ├── feature_engineering.ipynb
│   └── model_engineering.ipynb
├── src/
│   └── risk_score.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Module 1: Customer Churn Prediction

**Dataset:** Bank Customer Churn (Kaggle), 10,000 customer records

**Business context:** A bank with 10,000 customers and a 20.4% churn rate loses significant recurring revenue each year to customers it cannot identify in advance. This module builds a system to flag high-risk customers early and recommend retention actions before churn occurs.

### Pipeline

**Data Cleaning**

Removed identifier columns with no predictive value (RowNumber, CustomerId, Surname), encoded binary categorical variables, verified no null values or duplicates, and documented class imbalance at 79.6% retained vs 20.4% churned.

**Exploratory Data Analysis**

Analysed churn rates across all 14 features. Key findings:

| Feature | Finding |
|---|---|
| NumOfProducts | 2 products: 7.6% churn. 3 products: 82.7%. 4 products: 100% |
| Geography | Germany: 32.4% churn, vs 16.2% France and 16.7% Spain |
| Age | 50-59 band: 56% churn, highest of any segment |
| IsActiveMember | Inactive: 26.9% churn vs Active: 14.3% |
| Gender | Female: 25.1% churn vs Male: 16.5% |
| Complain | 99.5% churn rate, excluded from model as data leakage |
| HasCrCard, Card Type, Satisfaction Score, Salary | No meaningful signal, excluded |

**Feature Engineering**

Seven features derived from EDA observations:

| Feature | Logic |
|---|---|
| IsZeroBalance | 1 if Balance equals zero |
| AgeGroup | Age binned into five risk-based bands |
| IsHighRiskProduct | 1 if NumOfProducts is 3 or more |
| IsGermany | 1 if Geography is Germany |
| BalanceToSalaryRatio | Balance divided by EstimatedSalary |
| IsInactiveSenior | 1 if age 50 or above and inactive |
| IsFemaleGermany | 1 if female and Germany, highest observed interaction churn rate at 37.6% |

**Modelling**

XGBoost classifier trained inside a full sklearn Pipeline with OneHotEncoder (handle_unknown='ignore') and StandardScaler. SMOTE was evaluated and rejected as the base model produced higher recall, which is the priority metric for churn. Hyperparameters tuned via RandomizedSearchCV over 250 fits with 5-fold stratified cross-validation.

**Threshold Optimisation**

Default threshold of 0.5 replaced with a business cost-driven threshold of 0.35, derived by minimising total cost across false negatives (missed churners, cost Rs 10,000) and false positives (unnecessary retention calls, cost Rs 500).

**Explainability**

SHAP TreeExplainer applied to produce per-customer feature contribution scores. The Streamlit application renders these as bar charts and maps top risk drivers to specific retention actions.

**Experiment Tracking**

All runs logged to MLflow with parameters, metrics, and model artifacts stored in a local SQLite backend.

### Model Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.874 |
| Recall (churned class) | 81.6% |
| Precision (churned class) | 46.1% |
| F1-Score | 0.589 |
| Decision threshold | 0.35 |
| Cross-validation F1 | 0.619 (+/- 0.013) |

## Streamlit Application

A real-time churn risk tool built for bank operations teams. Adjust a customer profile in the sidebar and all outputs update instantly.

Features include churn probability with confidence interval and risk tier (Low, Medium, High, Critical), a SHAP chart explaining which features drove the prediction for that specific customer, personalised retention recommendations mapped from top SHAP contributors, a What-If Simulator that shows the probability shift when membership status or product count changes, and a peer benchmark showing historical churn rate among customers with a similar profile.

Live demo: https://huggingface.co/spaces/Bansal27/fintelliQ-churn

To run locally:

```bash
cd app
streamlit run app.py
```

## Rule-Based Risk Scorer

A separate interpretable scoring system (src/risk_score.py) assigns each customer a 0-100 risk score without using a trained model. Weights are derived directly from observed EDA churn rates and produce a RiskTier column (Low, Medium, High, Critical) for use in dashboards and retention queues.

This is intentionally separate from the XGBoost model and is designed for stakeholders who need transparent, auditable scoring without a machine learning dependency.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Machine learning | XGBoost, scikit-learn, imbalanced-learn |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Application | Streamlit |
| Data processing | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |


## Data

Dataset: [Bank Customer Churn Prediction, Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

Raw data files are not included in this repository. Download from Kaggle and place the CSV in the data/raw/ folder before running the notebooks.
