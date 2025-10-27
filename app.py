
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import shap
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Customer Churn & CLV Prediction",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("Customer Churn & CLV Prediction Dashboard")
st.markdown("An interactive dashboard to predict customer churn, analyze model performance, and explore Customer Lifetime Value (CLV).")

# --- Data Loading ---
@st.cache_data
def load_data():
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    clv_summary = pd.read_csv('data/processed/clv_summary.csv')
    log_reg_importance = pd.read_csv('reports/interpretability/log_reg_feature_importance.csv')
    return X_test, y_test, clv_summary, log_reg_importance

X_test, y_test, clv_summary, log_reg_importance = load_data()

# --- Model Loading ---
@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.joblib')
    log_reg = joblib.load('models/logistic_regression.joblib')
    
    # Load other models if they exist
    try:
        random_forest = joblib.load('models/random_forest.joblib')
    except FileNotFoundError:
        random_forest = None
    
    try:
        xgboost = joblib.load('models/xgboost.joblib')
    except FileNotFoundError:
        xgboost = None
        
    return scaler, log_reg, random_forest, xgboost

scaler, log_reg, random_forest, xgboost = load_models()

# --- Sidebar ---
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model for prediction and analysis:",
    ("Logistic Regression", "Random Forest", "XGBoost")
)

# --- Main App Tabs ---
tab1, tab2, tab3 = st.tabs(["Churn Prediction", "Model Performance", "CLV Overview"])

with tab1:
    st.header("🔮 Predict Customer Churn")
    st.write("Use the form below to get a churn prediction for a single customer.")

    # Input form
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1400.0)

    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col3:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])


    if st.button("Predict Churn"):
        # Create a dataframe from inputs
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Contract': [contract],
            'PaymentMethod': [payment_method],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'TechSupport': [tech_support],
            'PaperlessBilling': [paperless_billing],
            # Add default values for other columns that are not in the form
            'gender': [0], 
            'SeniorCitizen': [0],
            'Partner': [0],
            'Dependents': [0],
            'PhoneService': [1],
            'MultipleLines': [0],
            'OnlineBackup': [0],
            'DeviceProtection': [0],
            'StreamingTV': [0],
            'StreamingMovies': [0],
            'tenure_bucket': [0],
            'services_count': [0],
            'monthly_to_total_ratio': [0],
            'internet_no_techsupport': [0],
            'ExpectedTenure': [0],
            'CLV': [0]
        })

        # Preprocessing
        # One-hot encode categorical features
        input_data = pd.get_dummies(input_data, columns=['Contract', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 'TechSupport', 'PaperlessBilling'], drop_first=True)

        # Align columns with the training data
        input_data = input_data.reindex(columns=X_test.columns, fill_value=0)

        # Scale numerical features
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

        # --- Prediction ---
        if model_choice == "Logistic Regression":
            model = log_reg
        elif model_choice == "Random Forest":
            model = random_forest
        else:
            model = xgboost

        if model:
            prediction_proba = model.predict_proba(input_data)[:, 1][0]
            prediction = (prediction_proba > 0.5).astype(int)

            st.subheader("Prediction Result")
            churn_probability_percentage = prediction_proba * 100
            st.metric(label="Churn Probability", value=f"{churn_probability_percentage:.2f}%")

            if prediction == 1:
                st.error("Prediction: Customer is likely to CHURN.")
            else:
                st.success("Prediction: Customer is likely to STAY.")

            # --- SHAP Explanation ---
            st.subheader("Prediction Explanation (SHAP)")
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            
            # For binary classification, shap_values can be a list of two arrays
            # We are interested in the explanation for the positive class (churn)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            st.write("The plot below shows the contribution of each feature to the prediction.")
            fig, ax = plt.subplots()
            shap.force_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, shap_values, input_data, matplotlib=True, show=False)
            st.pyplot(fig, bbox_inches='tight')
            plt.close(fig)


        else:
            st.warning(f"{model_choice} model is not available.")

with tab2:
    st.header("📊 Model Performance Analysis")
    st.write(f"Showing performance metrics for the **{model_choice}** model.")
    
    st.write("Model performance analysis will be implemented here.")

with tab3:
    st.header("💡 Customer Lifetime Value (CLV) Overview")
    st.write("Insights into the distribution of CLV and its relationship with churn.")

    st.write("CLV overview will be implemented here.")

