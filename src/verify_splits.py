# src/verify_splits.py

import pandas as pd
import os

def verify_processed_data(path="data/processed"):
    """Check the integrity of processed data splits."""
    expected_files = [
        "X_train.csv", "X_val.csv", "X_test.csv",
        "y_train.csv", "y_val.csv", "y_test.csv"
    ]

    print("🔍 Verifying processed data files...\n")

    # 1. Check for existence
    for f in expected_files:
        file_path = os.path.join(path, f)
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {f}")
            return
        else:
            print(f"✅ Found: {f}")

    print("\n📊 Checking shapes and missing values...\n")

    # 2. Load datasets
    X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(path, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(path, "y_train.csv"))
    y_val = pd.read_csv(os.path.join(path, "y_val.csv"))
    y_test = pd.read_csv(os.path.join(path, "y_test.csv"))

    # 3. Print shapes
    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}   | y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

    # 4. Validate matching row counts
    assert len(X_train) == len(y_train), "❌ Mismatch: X_train and y_train row counts differ!"
    assert len(X_val) == len(y_val), "❌ Mismatch: X_val and y_val row counts differ!"
    assert len(X_test) == len(y_test), "❌ Mismatch: X_test and y_test row counts differ!"

    # 5. Check for missing values
    total_missing = (
        X_train.isnull().sum().sum() +
        X_val.isnull().sum().sum() +
        X_test.isnull().sum().sum()
    )
    if total_missing == 0:
        print("\n✅ No missing values detected in any feature split.")
    else:
        print(f"\n⚠️ Warning: There are {total_missing} missing values in the features.")

    print("\n🎯 Verification complete — all splits look good!")

if __name__ == "__main__":
    verify_processed_data()
