# Customer Churn Prediction (End-to-End ML Project)

## Business Problem
Customer churn is costly for subscription businesses. This project predicts churn risk and demonstrates how business levers like contract type and tenure affect customer retention.

## Dataset
IBM Telco Customer Churn dataset (~7,000 customers).

## Approach
- Exploratory Data Analysis (EDA)
- Feature engineering (tenure, early-tenure flag, service indicators)
- Logistic Regression baseline (ROC-AUC 0.86)
- Threshold tuning to maximize churn recall
- Random Forest comparison
- Explainability via coefficients
- Streamlit app for scenario testing

## Model Performance
- ROC-AUC: 0.86
- Churn Recall (tuned): ~74%
- Business-aligned decision threshold (0.35)

## Key Insights
- Month-to-month contracts drive the highest churn risk
- Early-tenure customers are most vulnerable
- Long-term contracts significantly reduce churn
- Support services lower churn probability

## Demo
Streamlit app allows interactive churn prediction and scenario testing.
