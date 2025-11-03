
import streamlit as st
import pandas as pd

def show_sidebar(X_train, X_test, y_test, clv_summary):
    """
    Displays the sidebar content for the Streamlit application.

    This includes the project title, description, key performance metrics (total customers,
    overall churn rate, average CLV), and credits.

    Args:
        X_train (pd.DataFrame): Training features DataFrame, used to calculate total customers and average CLV.
        X_test (pd.DataFrame): Testing features DataFrame, used to calculate total customers.
        y_test (pd.DataFrame): Testing target DataFrame, used to calculate overall churn rate.
        clv_summary (pd.DataFrame): DataFrame containing CLV summary information.
    """
    st.sidebar.title("📊 Customer Churn & CLV Prediction")
    st.sidebar.markdown("This application provides insights into customer churn and Customer Lifetime Value (CLV) using machine learning models.")

    st.sidebar.subheader("🔑 Key Metrics")
    total_customers = X_train.shape[0] + X_test.shape[0] # Assuming X_train and X_test represent the full customer base
    churn_rate = y_test['Churn'].mean() * 100 # Assuming 'Churn' column exists in y_test
    avg_clv = X_train['CLV'].mean()

    st.sidebar.metric("👥 Total Customers", f"{total_customers:,}")
    st.sidebar.metric("📉 Overall Churn Rate", f"{churn_rate:.2f}%")
    st.sidebar.metric("💰 Average CLV", f"${avg_clv:,.2f}")

    st.sidebar.subheader("🏆 Credits")
    st.sidebar.markdown("Developed by: Aghadiaye Amayanvbo Ernest")
    st.sidebar.markdown("GitHub: [aghadiayeamayanvboernest](https://github.com/aghadiayeamayanvboernest)")
    st.sidebar.write('LinkedIn: <a href="https://www.linkedin.com/in/ernest-jacob" target="_blank">Aghadiaye Ernest</a>', unsafe_allow_html=True)

    # Streamlit's built-in theme switcher is available in the settings menu (hamburger icon).
    # Custom theme toggles in the sidebar require more advanced techniques.
