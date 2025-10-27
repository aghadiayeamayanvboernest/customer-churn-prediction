# src/train_models.py
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load processed data
def load_data():
    base_path = "data/processed"
    X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(base_path, "y_train.csv")).values.ravel()
    X_val = pd.read_csv(os.path.join(base_path, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(base_path, "y_val.csv")).values.ravel()
    return X_train, X_val, y_train, y_val

# Train models and evaluate
def train_and_evaluate_models(X_train, y_train, X_val, y_val):
    # --- 1. Scale the data ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # --- 2. Handle class imbalance ---
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, scale_pos_weight=scale_pos_weight)
    }

    results = {}
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        # Train on scaled data
        model.fit(X_train_scaled, y_train)
        # Predict on scaled data
        y_pred = model.predict(X_val_scaled)

        metrics = {
            "Accuracy": accuracy_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred),
            "Recall": recall_score(y_val, y_pred),
            "F1": f1_score(y_val, y_pred),
            "ROC-AUC": roc_auc_score(y_val, y_pred)
        }

        results[name] = metrics
        print(f"{name} performance:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Also save the scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    print("\n✅ Scaler saved to models/scaler.joblib")

    return results, models

# Save all models
def save_all_models(models):
    os.makedirs("models", exist_ok=True)
    for name, model in models.items():
        model_path = f"models/{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(model, model_path)
        print(f"✅ Model saved: {name} → {model_path}")

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()
    results, models = train_and_evaluate_models(X_train, y_train, X_val, y_val)
    
    # Save all the trained models
    save_all_models(models)
    
    # Identify and print the best model based on ROC-AUC for informational purposes
    best_model_name = max(results, key=lambda x: results[x]["ROC-AUC"])
    print(f"\n🏆 Best model (based on ROC-AUC): {best_model_name}")