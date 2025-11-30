# Check the Streamlit App to try the models Here: 
https://diabetescheckup-hdugnkkzkmyhfaprt7sqjx.streamlit.app/

# 📘 About the Models
This app uses two machine learning models to help you understand your diabetes status:

## Diabetes Risk Estimator → predicts your numerical diabetes risk score.
Diabetes Diagnoser → predicts whether you're No Diabetes, Pre-Diabetes, or Type 2 Diabetes based on your inputs.
Diabetes Risk Estimator (Gradient Boosting Regressor)
The Diabetes Risk Estimator is built using a Gradient Boosting Regressor,
a model that combines many small decision trees to make a strong predictor.

### Model Performance

MSE (Mean Squared Error): 2.2601
MAE (Mean Absolute Error): 1.2084

### What These Metrics Mean:

- MSE measures how far predictions are from actual values on average, squared.
- Lower = better.
- An MSE of 2.26 means the model’s predictions are reasonably close to real risk scores.
- MAE measures the average absolute difference between predicted and actual scores.
  
-An MAE of 1.20 means predictions are usually within ±1.2 risk points of the true value.This makes the estimator reliable for screening, not medical diagnosis.

### Trained Features

Categorical Features:
gender, family_history_diabetes

Ordinal Features:
age_category, sleep_quality, diet_score_category

Numerical Features:
health_score, age*bmi

## Diabetes Diagnoser (Logistic Regression)
The Diabetes Diagnoser uses a Logistic Regression classifier,
a simple and highly interpretable model commonly used in medical research.

It predicts:

- No Diabetes
- Pre-Diabetes
- Type 2 Diabetes
- Model Performance
  
## 📊 Model Classification Report

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| No Diabetes   | 0.78      | 0.78   | 0.78     | 1406    |
| Pre-Diabetes  | 0.63      | 0.63   | 0.63     | 1394    |
| Type 2        | 0.82      | 0.82   | 0.82     | 1400    |
| **Accuracy**  | **0.74**  | —      | —        | 4200    |
| Macro Avg     | 0.74      | 0.74   | 0.74     | —       |
| Weighted Avg  | 0.74      | 0.74   | 0.74     | —       |

### Metric Interpretation (Simplified for Users)
1. Precision → When the model predicts a class, how correct is it?
2. Higher precision means fewer false alarms.
3. Recall → Of all real cases, how many did the model successfully detect?
4. Higher recall means fewer missed cases.
5. F1-Score → Balance of precision + recall.
6. Accuracy (0.74) → The model correctly predicts 74% of all cases.
- ✔ The model is strongest at detecting Type 2 Diabetes.
- ✔ It performs well on No Diabetes.
-  ⚠ It is moderately accurate for Pre-Diabetes, which is the hardest class to predict.

### Trained Features
postprandial_glucose_level (3 hours after meal)
diabetes_risk_score (output from the previous model)
Trainning Dataset https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset

⚠ Disclaimer:
This application uses synthetic (artificially generated) health data that reflect real data, taken from Kaggle. Use this App for educational, research, and self-screening purposes only.
It is not intended to diagnose, treat, or replace professional medical evaluation.
For any health concerns, please consult a licensed healthcare provider.
