
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)

# Page Configuration
st.set_page_config(page_title="Customer Churn & CLV Prediction", layout="wide")

# Helper functions
def create_speedometer(prob):
    risk_level = "Low"
    color = "green"
    if prob > 0.6:
        risk_level = "High"
        color = "red"
    elif prob > 0.3:
        risk_level = "Medium"
        color = "orange"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        number = {'suffix': '%', 'font': {'size': 50}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Churn Risk: {risk_level}", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'steps' : [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 60], 'color': 'lightyellow'},
                {'range': [60, 100], 'color': 'lightcoral'}],
        }
    ))
    return fig

def plot_log_reg_explanation(input_scaled, model, feature_names):
    contributions = input_scaled[0] * model.coef_[0]
    feature_contributions = pd.DataFrame({
        'feature': feature_names,
        'contribution': contributions
    }).sort_values(by='contribution', ascending=False)

    top_n = 10
    top_features = pd.concat([feature_contributions.head(top_n), feature_contributions.tail(top_n)])

    fig, ax = plt.subplots(figsize=(10, 6))
    top_features.plot(kind='barh', x='feature', y='contribution', ax=ax, 
                      color=(top_features['contribution'] > 0).map({True: 'r', False: 'g'}))
    ax.set_xlabel("Contribution to Churn Probability")
    ax.set_title("Top Feature Contributions for Logistic Regression")
    plt.tight_layout()
    return fig

# Caching data and models
@st.cache_data
def load_data():
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    clv_summary = pd.read_csv('data/processed/clv_summary.csv')
    return X_train, X_test, y_test, clv_summary

@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.joblib')
    log_reg = joblib.load('models/logistic_regression.joblib')
    random_forest = joblib.load('models/random_forest.joblib')
    xgboost = joblib.load('models/xgboost.joblib')
    return scaler, log_reg, random_forest, xgboost

@st.cache_resource
def get_tree_explainer(model):
    import shap
    return shap.TreeExplainer(model)

# Load data and models
X_train, X_test, y_test, clv_summary = load_data()
scaler, log_reg, random_forest, xgboost = load_models()
models = {"Logistic Regression": log_reg, "Random Forest": random_forest, "XGBoost": xgboost, "Ensemble": None}

from src.sidebar import show_sidebar

# Call the sidebar function
show_sidebar(X_train, X_test, y_test, clv_summary)
st.title("Customer Churn Prediction & Customer Lifetime Value (CLV)")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "💰 CLV Overview"])

with tab1:
    model_choice = st.selectbox("Select Model for Prediction", list(models.keys()))
    st.header(f"🔮 Predict Customer Churn with {model_choice}")
    st.write("Fill in the customer details to predict churn probability.")

    with st.form("prediction_form"):
        st.header("👤 Customer Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

        with col2:
            MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

        with col3:
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        col4, col5, col6 = st.columns(3)

        with col4:
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

        with col5:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1)
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        
        with col6:
            TotalCharges = st.number_input("Total Charges", min_value=0.0, value=100.0)
            SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        inputs = {
            'gender': gender, 'Partner': Partner, 'Dependents': Dependents, 'PhoneService': PhoneService,
            'MultipleLines': MultipleLines, 'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup, 'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport,
            'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies, 'Contract': Contract,
            'PaperlessBilling': PaperlessBilling, 'PaymentMethod': PaymentMethod, 'tenure': tenure,
            'MonthlyCharges': MonthlyCharges, 'TotalCharges': TotalCharges, 'SeniorCitizen': 1 if SeniorCitizen == 'Yes' else 0
        }
        input_df = pd.DataFrame([inputs])
        input_df_processed = pd.get_dummies(input_df, drop_first=True)

        # Feature Engineering: Interaction term
        if 'SeniorCitizen' in input_df_processed.columns and 'InternetService_Fiber optic' in input_df_processed.columns:
            input_df_processed['Senior_Fiber'] = input_df_processed['SeniorCitizen'] * input_df_processed['InternetService_Fiber optic']

        model_columns = scaler.get_feature_names_out()
        input_df_processed = input_df_processed.reindex(columns=model_columns, fill_value=0)
        input_scaled = scaler.transform(input_df_processed)
        
        if model_choice == "Ensemble":
            prediction_proba = np.mean([
                models["Logistic Regression"].predict_proba(input_scaled)[:, 1][0],
                models["Random Forest"].predict_proba(input_scaled)[:, 1][0],
                models["XGBoost"].predict_proba(input_scaled)[:, 1][0]
            ])
        else:
            selected_model = models[model_choice]
            prediction_proba = selected_model.predict_proba(input_scaled)[:, 1][0]

        st.subheader("🔮 Churn Prediction Result")
        
        # Visual Layout
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_speedometer(prediction_proba), use_container_width=True)

        with col2:
            risk_level = "Low"
            risk_color = "#28a745"
            if prediction_proba > 0.6:
                risk_level = "High"
                risk_color = "#dc3545"
            elif prediction_proba > 0.3:
                risk_level = "Medium"
                risk_color = "#ffc107"

            expected_tenure = 24
            clv = MonthlyCharges * expected_tenure
            st.metric(label="Estimated CLV", value=f"${clv:,.2f}")
            st.markdown(f'''
            <div style="
                background-color: {risk_color};
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;">
                <h3 style="color: white;">Churn Risk</h3>
                <p style="font-size: 24px; font-weight: bold;">{risk_level}</p>
            </div>
            ''', unsafe_allow_html=True)

        with st.expander("View Prediction Details"):
            st.subheader("🔍 Prediction Explanation")
            if model_choice == "Ensemble":
                st.write("Prediction explanation is not available for the Ensemble model.")
            elif model_choice in ['Random Forest', 'XGBoost']:
                explainer = get_tree_explainer(selected_model)
                shap_values = explainer.shap_values(input_scaled)
                st.write("The plot below shows the contribution of each feature to the prediction.")

                # Handle different shap_values structures
                if isinstance(shap_values, list):
                    # For scikit-learn models, shap_values is a list of two arrays
                    shap.force_plot(explainer.expected_value[1], shap_values[1], input_df_processed.iloc[0,:], matplotlib=True, show=False)
                else:
                    # For XGBoost, shap_values is a single array
                    shap.force_plot(explainer.expected_value, shap_values, input_df_processed.iloc[0,:], matplotlib=True, show=False)
                
                st.pyplot(bbox_inches='tight')
            else: # Logistic Regression
                st.write("For Logistic Regression, the chart below shows the top features influencing the prediction.")
                fig = plot_log_reg_explanation(input_scaled, selected_model, model_columns)
                st.pyplot(fig)

        st.subheader("💡 Business Insight & Recommendations")
        if prediction_proba > 0.6:
            insight_text = f"- **High Churn Risk:** This customer is at a high risk of churning. Proactive measures are strongly recommended.\n- **Key Factors:** The model indicates that the customer's short tenure of **{tenure} months** and their **{Contract}** contract are major contributors to this risk.\n- **Recommendation:** Prioritize this customer for a retention campaign. Consider offering a loyalty discount, a contract upgrade to a one or two-year term to increase stability, or a bundle with additional services like Tech Support, especially if they are using Fiber optic internet."
        elif prediction_proba > 0.3:
            insight_text = f"- **Medium Churn Risk:** This customer shows some signs of potential churn. It would be wise to monitor their account and engage them proactively.\n- **Key Factors:** The customer's current service combination and contract type may not be optimal. \n- **Recommendation:** Engage this customer with a satisfaction survey to identify any pain points. A personalized check-in from customer support could also be beneficial. Look for opportunities to enhance their service package to increase value and loyalty."
        else:
            insight_text = f"- **Low Churn Risk:** This customer is likely to remain loyal.\n- **Recommendation:** Continue to provide excellent service. This customer could be a good candidate for upselling new services or features, given their low churn risk."
        st.markdown(insight_text)

from src.tabs.model_performance_tab import show_model_performance_tab

with tab2:
    show_model_performance_tab(models, scaler, X_test, y_test)

from src.tabs.clv_overview_tab import show_clv_overview_tab

with tab3:
    show_clv_overview_tab(X_train)

