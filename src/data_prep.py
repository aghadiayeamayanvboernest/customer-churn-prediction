# src/data_prep.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_data(path="data/raw/telco.csv"):
    """Load raw dataset"""
    df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df):
    """Handle TotalCharges missing values and convert data types"""
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"], inplace=True)
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    return df

def engineer_features(df):
    """Create business-driven features"""
    # tenure_bucket
    df["tenure_bucket"] = pd.cut(df["tenure"],
                                 bins=[0, 6, 12, 24, 1000],
                                 labels=["0-6m", "6-12m", "12-24m", "24m+"])

    # services_count (Corrected Logic)
    service_cols = ["PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    # Count services that are not 'No' or 'No internet service' or 'No phone service'
    df["services_count"] = df[service_cols].apply(lambda x: x[~x.isin(["No", "No internet service", "No phone service"])].count(), axis=1)

    # monthly_to_total_ratio
    df["monthly_to_total_ratio"] = df["TotalCharges"] / np.maximum(1, df["tenure"] * df["MonthlyCharges"])

    # flag: internet but no tech support
    df["internet_no_techsupport"] = ((df["InternetService"] != "No") & (df["TechSupport"] == "No")).astype(int)

    # Expected tenure assumption (simplified)
    df["ExpectedTenure"] = np.where(df["Contract"] == "Month-to-month", 12,
                            np.where(df["Contract"] == "One year", 24, 36))

    # CLV calculation
    df["CLV"] = df["MonthlyCharges"] * df["ExpectedTenure"]

    # Separate target variable
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df

def split_data(df, target_col="Churn"):
    """Split into train/val/test (60/20/20) before encoding"""
    X = df.drop(columns=[target_col, "customerID"]) # Drop customerID as it's an identifier
    y = df[target_col]

    # Split into train (60%) and temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)

    # Split temp into validation (50% of 40% = 20%) and test (50% of 40% = 20%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def encode_data(X_train, X_val, X_test):
    """Encode categorical features after splitting to prevent data leakage."""
    # Identify categorical columns (object type)
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on training data only
        X_train[col] = le.fit_transform(X_train[col])
        # Transform validation and test data
        X_val[col] = le.transform(X_val[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le

    print("✅ Categorical features encoded.")
    return X_train, X_val, X_test, label_encoders

def save_splits(output_dir, **kwargs):
    """Save processed dataframes to a directory."""
    os.makedirs(output_dir, exist_ok=True)
    for name, df in kwargs.items():
        df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    print(f"✅ Processed splits saved in {output_dir}")

if __name__ == "__main__":
    # 1. Load and Clean
    df = load_data()
    df = clean_data(df)

    # 2. Engineer Features (and separate target)
    df = engineer_features(df)

    # 3. Split Data (before encoding)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # 4. Encode Categorical Features (fit on train, transform others)
    X_train, X_val, X_test, encoders = encode_data(X_train, X_val, X_test)

    # 5. Save Splits
    save_splits("data/processed",
                X_train=X_train, X_val=X_val, X_test=X_test,
                y_train=pd.DataFrame(y_train), y_val=pd.DataFrame(y_val), y_test=pd.DataFrame(y_test))
