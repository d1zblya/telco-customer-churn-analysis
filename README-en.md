# Telco Customer Churn Prediction

## Description

This project focuses on predicting customer churn for a telecom company using historical client data.

**Goal** — build a model that determines whether a customer will leave the company in the near future based on their characteristics, behavior, and service usage.

The project is implemented in a Jupyter Notebook and covers the full machine learning pipeline — from data analysis to model evaluation and result interpretation.

## Objective & Task

Telecom companies lose significant revenue due to customer churn. The churn prediction task helps:

- identify at-risk customers in advance
- personalize retention offers
- reduce marketing costs and improve retention

**Task type:** binary classification  
**Target variable:** `Churn` (1 — churned, 0 — stayed)

## Data

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Feature groups:**

- **Demographics:** gender, age, marital status
- **Services:** connection type (DSL, Fiber optic), add-ons (TV, internet security, tech support, etc.)
- **Account info:** tenure, monthly and total charges
- **Target:** churn status (`Churn`)

**Preprocessing steps:**

- missing values & duplicates check
- distribution and outlier analysis
- initial statistical overview

## EDA (Exploratory Data Analysis)

The exploratory analysis examined:

- distributions of key numerical and categorical features
- relationships between features and the target
- correlations between variables
- impact of tenure and spending levels on churn probability

**Key EDA insights:**

- short tenure (< 12 months) strongly correlates with churn
- customers with high monthly charges are more likely to leave

## Data Preprocessing

**Steps included:**

- handling missing values
- encoding categorical variables via `OneHotEncoder`
- scaling numerical features (where required)
- train-test split (90/10)

## Model Training

**Algorithms tested:**

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier

**For each model:**

- hyperparameter tuning (`GridSearchCV`)
- evaluation using accuracy, precision, recall, F1, and ROC AUC
- confusion matrix analysis

## Results & Metrics

Best performance for class **'0' (Stayed)** shown by **KNN** and **AdaBoost**:

| Model              | Accuracy | F1-score |
|--------------------|----------|----------|
| Logistic Regression| 0.73     | 0.81     |
| **KNN**            | **0.81** | **0.88** |
| SVC                | 0.73     | 0.81     |
| Decision Tree      | 0.73     | 0.81     |
| Random Forest      | 0.72     | 0.80     |
| **AdaBoost**       | **0.82** | **0.89** |

**KNN** and **AdaBoost** demonstrated the best balance of accuracy and generalization.