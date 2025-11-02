# Customer Churn Prediction & Customer Lifetime Value (CLV) Dashboard

## Project Overview
This project implements an interactive Streamlit dashboard for predicting customer churn and analyzing Customer Lifetime Value (CLV). It leverages machine learning models to provide insights into customer behavior, identify at-risk customers, and offer business recommendations to improve retention and maximize customer value.

## Business Framing
This project is designed to help a telecommunications company reduce customer churn and increase profitability. By predicting which customers are likely to churn, the company can proactively target them with retention campaigns. The dashboard also provides insights into Customer Lifetime Value (CLV), allowing the business to segment customers and prioritize high-value customers.

## Features
The dashboard is structured into several key sections:

*   **🔮 Predict Tab:**
    *   Allows users to input customer details and get real-time churn probability predictions using a selected machine learning model.
    *   Visualizes churn risk with a speedometer gauge.
    *   Provides estimated Customer Lifetime Value (CLV) for the input customer.
    *   Offers detailed prediction explanations (SHAP values for tree-based models, feature coefficients for Logistic Regression).
    *   Generates actionable business insights and recommendations based on the predicted churn risk.

*   **📊 Model Performance Tab:**
    *   Compares the performance of different machine learning models (Logistic Regression, Random Forest, XGBoost) based on metrics like Precision, Recall, F1-Score, and AUC-ROC.
    *   Highlights the best-performing model for each metric.
    *   Displays ROC curves for all models.
    *   Provides detailed analysis for a selected model, including its Confusion Matrix and Feature Importance visualizations.

*   **💰 CLV Overview Tab:**
    *   Presents an overview of Customer Lifetime Value distribution.
    *   Visualizes churn rate by CLV segments.
    *   Offers key business insights derived from CLV analysis.

*   **Sidebar:**
    *   Displays a project title and a brief description.
    *   Shows key overall metrics: Total Customers, Overall Churn Rate, and Average CLV.
    *   Includes a "Credits" section.

## CLV Assumptions
The Customer Lifetime Value (CLV) in this dashboard is a simplified estimation. It is calculated as:
`CLV = MonthlyCharges * Expected_Tenure`
The `Expected_Tenure` is a fixed value of 24 months. This is a simplification and in a real-world scenario, it would be more accurately predicted based on customer data and other factors.

## Technologies Used
*   **Python:** Programming language
*   **Streamlit:** For building the interactive web application
*   **Pandas & NumPy:** For data manipulation and numerical operations
*   **Scikit-learn:** For machine learning model training and evaluation
*   **XGBoost:** Gradient Boosting library
*   **SHAP:** For model interpretability (SHapley Additive exPlanations)
*   **Matplotlib, Seaborn, Plotly:** For data visualization
*   **Joblib:** For saving and loading trained models and scaler

## Installation
To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aghadiayeamayanvboernest/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the Streamlit application, execute the following command from the project's root directory:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Project Structure
```
.
├── app.py                      # Main Streamlit application file
├── requirements.txt            # Python dependencies
├── data/
│   ├── processed/              # Processed data files (X_train, y_test, clv_summary, etc.)
│   └── raw/                    # Raw data files (telco.csv)
├── models/                     # Trained machine learning models and scaler
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── scaler.joblib
│   └── xgboost.joblib
├── notebooks/                  # Jupyter notebooks for EDA, model training, etc.
├── reports/
│   ├── figures/                # Visualizations and plots
│   ├── insights/               # Text-based insights
│   └── interpretability/       # Model interpretability plots
└── src/
    ├── clv_analysis.py         # Functions for CLV analysis
    ├── data_prep.py            # Data preprocessing scripts
    ├── interpretability.py     # Scripts for model interpretability
    ├── predict.py              # Prediction logic
    ├── train_models.py         # Model training scripts
    ├── verify_splits.py        # Data split verification
    ├── sidebar.py              # Sidebar content and logic
    └── tabs/
        ├── clv_overview_tab.py # CLV Overview tab content
        └── model_performance_tab.py # Model Performance tab content
```

## Data
The project utilizes a Telco Customer Churn dataset, commonly available from sources like Kaggle or IBM. This dataset includes various demographic, service, and account information for customers. The data is preprocessed and split into training, validation, and test sets.

**Acknowledgment:** The Telco Customer Churn dataset is publicly available and widely used for educational and research purposes. We acknowledge the original creators and distributors of this valuable dataset.

## Models
The dashboard integrates three machine learning models for churn prediction:
*   **Logistic Regression:** A linear model providing interpretable coefficients.
*   **Random Forest:** An ensemble tree-based model known for its robustness.
*   **XGBoost:** A highly efficient and powerful gradient boosting framework.

## Credits
*   **Developed by:** Aghadiaye Amayanvbo Ernest
*   **GitHub:** [https://github.com/aghadiayeamayanvboernest](https://github.com/aghadiayeamayanvboernest)
*   **Assisted by:** Gemini CLI Agent

## License
This project is open-source and available under the MIT License.