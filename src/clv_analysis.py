# src/clv_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def load_processed_data(path="data/processed/X_train.csv", y_path="data/processed/y_train.csv"):
    """Load processed feature and target data for CLV analysis."""
    X = pd.read_csv(path)
    y = pd.read_csv(y_path)
    X["Churn"] = y.values
    print(f"✅ Loaded processed data: {X.shape[0]} rows, {X.shape[1]} columns")
    return X

def analyze_clv(df):
    """Compute CLV quartiles and churn rates."""
    # Ensure CLV exists
    if "CLV" not in df.columns:
        raise ValueError("❌ 'CLV' column not found. Make sure data_prep.py created it.")

    # Create quartiles
    df["CLV_quartile"] = pd.qcut(df["CLV"], 4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])

    # Compute churn rate by quartile
    clv_summary = df.groupby("CLV_quartile")["Churn"].agg(["mean", "count"]).reset_index()
    clv_summary.rename(columns={"mean": "churn_rate", "count": "customer_count"}, inplace=True)

    print("\n📊 CLV vs Churn Summary:")
    print(clv_summary)

    # Save summary
    os.makedirs("data/processed", exist_ok=True)
    clv_summary.to_csv("data/processed/clv_summary.csv", index=False)
    print("\n✅ CLV summary saved to data/processed/clv_summary.csv")

    return clv_summary

def plot_clv_distribution(df):
    """Plot CLV distribution."""
    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(df["CLV"], bins=30, color="skyblue", edgecolor="black")
    plt.title("Customer Lifetime Value (CLV) Distribution")
    plt.xlabel("CLV ($)")
    plt.ylabel("Number of Customers")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/clv_distribution.png")
    plt.close()
    print("📈 Saved: reports/figures/clv_distribution.png")

def plot_churn_by_clv_segment(clv_summary):
    """Plot churn rate by CLV quartile."""
    plt.figure(figsize=(7, 5))
    plt.bar(clv_summary["CLV_quartile"], clv_summary["churn_rate"], color="orange", edgecolor="black")
    plt.title("Churn Rate by CLV Quartile")
    plt.xlabel("CLV Segment")
    plt.ylabel("Churn Rate")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/churn_rate_by_clv.png")
    plt.close()
    print("📊 Saved: reports/figures/churn_rate_by_clv.png")

def generate_insights(clv_summary):
    """Generate and save more detailed, data-driven insights."""
    insights = []

    # Find the segment with the highest churn rate
    highest_churn_segment = clv_summary.loc[clv_summary['churn_rate'].idxmax()]
    
    # Get churn rates for the highest and lowest CLV quartiles
    high_clv_churn = clv_summary[clv_summary['CLV_quartile'] == 'Q4 (High)']['churn_rate'].iloc[0]
    low_clv_churn = clv_summary[clv_summary['CLV_quartile'] == 'Q1 (Low)']['churn_rate'].iloc[0]

    # --- Generate Insights ---
    insights.append("--- Key Insights ---")
    if high_clv_churn < low_clv_churn:
        insights.append(f"💡 Loyalty Insight: High-CLV customers (Q4) are the most loyal, with a churn rate of only {high_clv_churn:.2%}.")
    else:
        insights.append(f"⚠️ Retention Risk: High-CLV customers (Q4) are churning at a concerning rate of {high_clv_churn:.2%}.")

    insights.append(f"🔥 Highest Churn Risk: The **{highest_churn_segment['CLV_quartile']}** segment has the highest churn rate at a staggering **{highest_churn_segment['churn_rate']:.2%}**.")

    # --- Generate Recommendations ---
    insights.append("\n--- Recommendations ---")
    insights.append(f"1. **Urgent Action:** Prioritize retention efforts on the **{highest_churn_segment['CLV_quartile']}** segment. Their high churn rate presents the biggest immediate threat to revenue.")
    insights.append("2. **Nurture Loyalty:** While Q4 customers are loyal, their high value warrants continued engagement and satisfaction monitoring to maintain low churn.")
    
    # --- Add overall average churn ---
    avg_churn = clv_summary['churn_rate'].mean()
    insights.append(f"\n📊 Overall average churn rate across all segments: {avg_churn:.2%}")

    # --- Save Insights ---
    os.makedirs("reports/insights", exist_ok=True)
    with open("reports/insights/clv_insights.txt", "w") as f:
        for line in insights:
            f.write(line + "\n")

    print("\n🧠 Detailed insights generated and saved to reports/insights/clv_insights.txt")

if __name__ == "__main__":
    df = load_processed_data()
    clv_summary = analyze_clv(df)
    plot_clv_distribution(df)
    plot_churn_by_clv_segment(clv_summary)
    generate_insights(clv_summary)
    print("\n✅ CLV Analysis complete!")
