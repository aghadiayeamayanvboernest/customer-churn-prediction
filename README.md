# Customer Churn Prediction & Customer Lifetime Value (CLV)

This project aims to predict customer churn for a SaaS company and identify high-value customers to prioritize for retention efforts. By analyzing customer data, we build machine learning models to predict churn probability and calculate Customer Lifetime Value (CLV). The entire analysis is presented in an interactive Streamlit application.

**Video Demo:** [Link to your 2-3 minute video demo]

## Business Context

Customer churn is a significant challenge for SaaS businesses, with an average of 5-7% of annual revenue lost. This project addresses this by:
1.  **Predicting Churn:** Identifying customers who are likely to cancel their subscriptions.
2.  **Estimating Customer Value:** Using CLV to quantify the financial value of each customer, enabling targeted retention strategies.

## Features

*   **Data Preparation:** Cleans and preprocesses the IBM Telco Customer Churn dataset, including business-driven feature engineering.
*   **CLV Analysis:** Calculates CLV for each customer and segments them into value-based quartiles (Low, Medium, High, Premium).
*   **Machine Learning Models:** Trains and evaluates three different models for churn prediction:
    *   Logistic Regression (Baseline)
    *   Random Forest
    *   XGBoost
*   **Model Interpretability:** Uses SHAP (SHapley Additive exPlanations) to explain model predictions, providing both global and local feature importance.
*   **Interactive Application:** A single-page Streamlit app with tabs for:
    *   **Predict:** Get real-time churn predictions and explanations for a single customer.
    *   **Model Performance:** Compare evaluation metrics, ROC curves, and feature importance for all models.
    *   **CLV Overview:** Visualize CLV distribution and churn rates across different customer segments.
*   **Cloud Deployment:** The application is deployed on Streamlit Community Cloud for easy access.

## Repository Structure

```
project2-churn-prediction/
├── README.md
├── AI_USAGE.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_prep.py
│   ├── clv_analysis.py
│   ├── train_models.py
│   ├── interpretability.py
│   └── predict.py
├── models/
│   ├── logistic.pkl
│   ├── rf.pkl
│   └── xgb.pkl
├── app.py
└── notebooks/
    └── exploration.ipynb (optional)
```

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd customer-churn-prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the data preparation and model training scripts:**
    *This step is necessary to generate the processed data and trained models.*
    ```bash
    python src/data_prep.py
    python src/train_models.py
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will be available at `http://localhost:8501`.

## CLV Assumptions

The Customer Lifetime Value (CLV) is calculated with the following formula and assumptions:

*   **Formula:** `CLV = MonthlyCharges × ExpectedTenure`
*   **Expected Tenure:** A predefined assumption for how long a customer is expected to stay. This should be clearly stated and justified based on business knowledge or data analysis. For this project, the assumption for `ExpectedTenure` needs to be defined.

## Data Source

The project uses the **IBM Telco Customer Churn** dataset.
*   **Direct CSV Link:** [https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

## Dependencies

The main libraries used in this project are listed in `requirements.txt`. Key dependencies include:

*   `pandas`
*   `scikit-learn`
*   `xgboost`
*   `shap`
*   `streamlit`
*   `matplotlib`
