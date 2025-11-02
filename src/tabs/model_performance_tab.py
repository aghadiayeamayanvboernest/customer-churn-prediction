
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns

def show_model_performance_tab(models, scaler, X_test, y_test):
    """
    Displays the Model Performance tab content.

    This tab provides a comparison of different machine learning models based on various
    performance metrics, ROC curves, confusion matrices, and feature importance plots.

    Args:
        models (dict): A dictionary of trained machine learning models.
        scaler: The fitted scaler object used for data transformation.
        X_test (pd.DataFrame): Testing features DataFrame.
        y_test (pd.DataFrame): Testing target DataFrame.
    """
    st.header("📊 Model Performance")
    st.write("Comparing the performance of different models.")

    X_test_scaled = scaler.transform(X_test)
    performance_data = []
    for name, model in models.items():
        if model is None:
            continue
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        performance_data.append({
            "Model": name,
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "AUC-ROC": roc_auc_score(y_test, y_proba)
        })
    
    performance_df = pd.DataFrame(performance_data)

    st.subheader("📈 Performance Metrics Table")
    st.dataframe(performance_df, use_container_width=True)

    st.header("🏆 Model Performance Summary")
    st.write("A quick overview of the best performing models for each metric.")

    best_precision_model = performance_df.loc[performance_df['Precision'].idxmax()]
    best_recall_model = performance_df.loc[performance_df['Recall'].idxmax()]
    best_f1_model = performance_df.loc[performance_df['F1-Score'].idxmax()]
    best_auc_model = performance_df.loc[performance_df['AUC-ROC'].idxmax()]

    card_style = """
    <style>
    .card {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .card h3 {
        color: white;
    }
    .card p {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'''
        <div class="card" style="background-color: #007bff;">
            <h3>Best Precision</h3>
            <p>{best_precision_model['Model']}</p>
            <p>{best_precision_model['Precision']:.4f}</p>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''
        <div class="card" style="background-color: #28a745;">
            <h3>Best Recall</h3>
            <p>{best_recall_model['Model']}</p>
            <p>{best_recall_model['Recall']:.4f}</p>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''
        <div class="card" style="background-color: #ffc107;">
            <h3>Best F1-Score</h3>
            <p>{best_f1_model['Model']}</p>
            <p>{best_f1_model['F1-Score']:.4f}</p>
        </div>
        ''', unsafe_allow_html=True)

    with col4:
        st.markdown(f'''
        <div class="card" style="background-color: #dc3545;">
            <h3>Best AUC-ROC</h3>
            <p>{best_auc_model['Model']}</p>
            <p>{best_auc_model['AUC-ROC']:.4f}</p>
        </div>
        ''', unsafe_allow_html=True)

    st.subheader(f"🏅 Overall Best Model (based on F1-Score): {best_f1_model['Model']}")

    st.subheader("📉 ROC Curves for All Models")
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, model in models.items():
        if model is None:
            continue
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, label=name)
    
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    ax.legend()
    st.pyplot(fig)

    st.header("🔍 Detailed Model Analysis")
    st.write("""
    <p style='font-size: 16px;'>
    Select a model from the dropdown to view its confusion matrix and feature importance.
    </p>
    """, unsafe_allow_html=True)
    model_choice = st.selectbox("Select a model to view its Confusion Matrix and Feature Importance:", [m for m in models.keys() if m != 'Ensemble'])

    st.subheader(f"📈 Confusion Matrix: {model_choice}")
    selected_model = models[model_choice]
    y_pred_selected = selected_model.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, y_pred_selected)
    
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix: {model_choice}')
    st.pyplot(fig)

    st.subheader(f"💡 Feature Importance: {model_choice}")
    st.write("""
    <p style='font-size: 16px;'>
    This section visualizes how different features contribute to the model's predictions.
    </p>
    """, unsafe_allow_html=True)

    if model_choice == 'Logistic Regression':
        st.write("""
        <p style='font-size: 14px;'>
        For Logistic Regression, the plot shows the coefficients of the features. 
        Positive coefficients indicate features that increase the likelihood of churn, 
        while negative coefficients indicate features that decrease it. The magnitude 
        of the coefficient reflects the strength of the influence.
        </p>
        """, unsafe_allow_html=True)
        st.image('reports/interpretability/log_reg_feature_importance.png', width=800)
    elif model_choice == 'Random Forest':
        st.write("""
        <p style='font-size: 14px;'>
        For tree-based models like Random Forest and XGBoost, SHAP (SHapley Additive exPlanations) 
        values are used to explain individual predictions. The summary plot below shows 
        which features are most important across the dataset. Each dot represents a customer, 
        its position on the x-axis shows the SHAP value (impact on model output), and 
        its color indicates the feature value (e.g., red for high, blue for low).
        </p>
        """, unsafe_allow_html=True)
        st.image('reports/interpretability/shap_summary_random_forest.png', width=800)
    else:
        st.write("""
        <p style='font-size: 14px;'>
        Similar to Random Forest, for XGBoost, SHAP values are used. The plot illustrates 
        the distribution of SHAP values for each feature, indicating their overall importance 
        and direction of impact on the model's output (churn probability).
        </p>
        """, unsafe_allow_html=True)
        st.image('reports/interpretability/shap_summary_xgboost.png', width=800)
