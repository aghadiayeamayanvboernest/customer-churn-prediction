"""Interpretability Analysis for Churn Prediction Models
-------------------------------
This script explains the trained churn prediction models using:
- Logistic Regression coefficients
- SHAP values for tree models (Random Forest & XGBoost)
Outputs:
    - Feature importance plots
    - SHAP summary plots
    - Text insights saved in reports/insights/
"""

import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# === PATHS ===
MODEL_DIR = "models"
REPORTS_DIR = "reports/interpretability"
DATA_PATH = "data/processed/X_test.csv"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs("reports/insights", exist_ok=True)

# === Load Test Data ===
X_test = pd.read_csv(DATA_PATH)

# === Load Models ===
log_model_path = os.path.join(MODEL_DIR, "logistic_regression.joblib")
rf_model_path = os.path.join(MODEL_DIR, "random_forest.joblib")
xgb_model_path = os.path.join(MODEL_DIR, "xgboost.joblib")

models = {}

if os.path.exists(log_model_path):
    models["Logistic Regression"] = joblib.load(log_model_path)
if os.path.exists(rf_model_path):
    models["Random Forest"] = joblib.load(rf_model_path)
if os.path.exists(xgb_model_path):
    models["XGBoost"] = joblib.load(xgb_model_path)

print(f"✅ Loaded models: {', '.join(models.keys())}\n")


# === Logistic Regression Feature Importance ===
def explain_logistic_regression(model, X):
    """
    Generates and saves feature importance for a Logistic Regression model.

    Calculates coefficients, saves them to a CSV, and plots the top 15 influential features.

    Args:
        model (LogisticRegression): The trained Logistic Regression model.
        X (pd.DataFrame): The DataFrame of features used for explanation.
    """
    coef = model.coef_[0]
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": coef
    }).sort_values("importance", ascending=False)

    # Save importance as CSV
    feature_importance.to_csv(os.path.join(REPORTS_DIR, "log_reg_feature_importance.csv"), index=False)

    # Plot top features
    plt.figure(figsize=(8, 6))
    feature_importance.head(15).set_index("feature").plot(kind="barh", legend=False)
    plt.title("Top 15 Influential Features - Logistic Regression")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "log_reg_feature_importance.png"))
    plt.close()

    print("🧠 Logistic Regression coefficients saved and plotted.\n")


# === SHAP Analysis for Tree Models ===
def explain_tree_model(model, X, model_name):
    """
    Performs SHAP analysis for tree-based models and saves summary plots.

    Generates SHAP summary plots (dot and bar) to explain feature contributions.

    Args:
        model: The trained tree-based model (e.g., RandomForestClassifier, XGBClassifier).
        X (pd.DataFrame): The DataFrame of features used for explanation.
        model_name (str): The name of the model (e.g., "Random Forest", "XGBoost").
    """
    print(f"🔍 Explaining {model_name} with SHAP...")

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Summary Plot
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Summary - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"shap_summary_{model_name.lower().replace(' ', '_')}.png"))
    plt.close()

    # Bar plot
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"shap_bar_{model_name.lower().replace(' ', '_')}.png"))
    plt.close()

    print(f"✅ SHAP plots saved for {model_name}.\n")


# === Generate Insights ===
def save_interpretability_insights():
    """
    Saves general interpretability insights to a text file.

    Provides a summary of how model interpretability is approached in the project.
    """
    insights = [
        "--- Model Interpretability Insights ---",
        "1. Logistic Regression coefficients reveal which features most increase or decrease churn likelihood.",
        "2. SHAP values highlight non-linear feature effects in Random Forest and XGBoost models.",
        "3. Positive SHAP or coefficient values → higher churn risk; negative values → retention signals.",
        "4. Use these insights in Streamlit to show users what drives churn predictions for each customer."
    ]

    with open("reports/insights/model_interpretability_insights.txt", "w") as f:
        for line in insights:
            f.write(line + "\n")

    print("📝 Interpretability insights saved to reports/insights/model_interpretability.txt\n")


# === Run Explanations ===
for name, model in models.items():
    if isinstance(model, LogisticRegression):
        explain_logistic_regression(model, X_test)
    else:
        explain_tree_model(model, X_test, name)

save_interpretability_insights()

print("🎯 Interpretability analysis complete! Check the 'reports/interpretability/' folder for plots.")