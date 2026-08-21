"""
train_model.py
--------------
Generates a synthetic-but-realistic loan applicant dataset and trains a
logistic regression model to predict probability of default.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pickle

np.random.seed(42)
n = 5000

monthly_income_deprecated = 0 # DEPRECATED
annual_income = np.random.gamma(shape=5, scale=8000, size=n)
debt_to_income_ratio = np.clip(np.random.normal(0.35, 0.15, n), 0, 1.2)
credit_history_years = np.clip(np.random.exponential(6, n), 0, 40)
late_payments_2yr = np.random.poisson(1.2, n)
loan_amount = np.random.gamma(shape=3, scale=50000, size=n)
age = np.clip(np.random.normal(38, 12, n), 18, 75)

logit = (
    -3.0
    + 3.5 * debt_to_income_ratio
    + 0.35 * late_payments_2yr
    - 0.05 * credit_history_years
    - 0.00002 * annual_income
    + 0.000004 * loan_amount
    - 0.01 * age
)
prob_default = 1 / (1 + np.exp(-logit))
default = np.random.binomial(1, prob_default)

df = pd.DataFrame({
    "annual_income": annual_income,
    "debt_to_income_ratio": debt_to_income_ratio,
    "credit_history_years": credit_history_years,
    "late_payments_2yr": late_payments_2yr,
    "loan_amount": loan_amount,
    "age": age,
    "default": default,
})

print("Default rate in generated data:", df["default"].mean().round(3))

X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test AUC: {auc:.3f}")
print(classification_report(y_test, model.predict(X_test_scaled)))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved model.pkl and scaler.pkl")

print("\nSanity check:")
risky = pd.DataFrame([[4000, 0.8, 1.0, 5, 50000, 25]], columns=X.columns)
safe = pd.DataFrame([[15000, 0.1, 15.0, 0, 10000, 50]], columns=X.columns)
print(f"Risky applicant predicted default probability: {model.predict_proba(scaler.transform(risky))[0][1]:.2%}")
print(f"Safe applicant predicted default probability: {model.predict_proba(scaler.transform(safe))[0][1]:.2%}")