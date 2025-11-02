
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_clv_overview_tab(X_train):
    """
    Displays the Customer Lifetime Value (CLV) Overview tab content.

    This tab provides visualizations and insights related to CLV distribution,
    churn rate by CLV segment, and key business insights.

    Args:
        X_train (pd.DataFrame): Training features DataFrame, expected to contain 'CLV' column.
    """
    st.header("💰 Customer Lifetime Value (CLV) Overview")
    st.write("Understanding customer value and churn patterns.")

    # CLV Distribution
    st.subheader("📈 CLV Distribution")
    fig, ax = plt.subplots()
    ax.hist(X_train['CLV'], bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel('CLV')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Distribution of Customer Lifetime Value')
    st.pyplot(fig, use_container_width=True)

    st.subheader("📊 Churn Rate by CLV Segment")
    st.image('reports/figures/churn_rate_by_clv.png')

    st.subheader("💡 Key Business Insights")
    with open('reports/insights/clv_insights.txt', 'r') as f:
        insights = f.read()
    st.write(insights)
