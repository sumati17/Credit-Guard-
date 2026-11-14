
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CreditGuard | Credit Risk Intelligence",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CreditGuard")
st.subheader("Credit Risk Intelligence Dashboard")

st.write(
    "An ML-powered prototype for estimating customer "
    "delinquency risk and suggesting responsible support."
)

st.info(
    "Educational prototype only. Predictions are not validated "
    "for real lending decisions. Risk thresholds are provisional."
)


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# --------------------------------------------------
# 3. Input form
# --------------------------------------------------

st.header("Customer Financial Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=500000.0,
            step=10000.0
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=0.0,
            max_value=1000.0,
            value=700.0
        )

        credit_utilization = st.number_input(
            "Credit Utilization",
            min_value=0.0,
            max_value=1.5,
            value=0.45,
            step=0.01,
            help="Enter the same ratio scale used in the training dataset."
        )

        missed_payments = st.number_input(
            "Recorded Missed Payments",
            min_value=0,
            value=1,
            step=1
        )

    with col2:
        loan_balance = st.number_input(
            "Loan Balance",
            min_value=0.0,
            value=100000.0,
            step=5000.0
        )

        debt_to_income = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            max_value=10.0,
            value=0.30,
            step=0.01
        )

        account_tenure = st.number_input(
            "Account Tenure (years)",
            min_value=0.0,
            value=5.0,
            step=0.5
        )

    st.subheader("Monthly Payment History")

    st.caption(
        "Select the payment status for each of the six recorded months."
    )

    payment_options = ["On-time", "Late", "Missed"]

    payment_cols = st.columns(3)

    payment_statuses = []

    for i in range(6):
        with payment_cols[i % 3]:
            status = st.selectbox(
                f"Month {i + 1}",
                payment_options,
                key=f"month_{i + 1}"
            )
            payment_statuses.append(status)

    submitted = st.form_submit_button(
        "Predict Delinquency Risk",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------

if submitted:

    payment_mapping = {
        "On-time": 0,
        "Late": 1,
        "Missed": 2
    }

    payment_values = [
        payment_mapping[status]
        for status in payment_statuses
    ]

    payment_array = np.array(payment_values)

    on_time_count = int(np.sum(payment_array == 0))
    late_count = int(np.sum(payment_array == 1))
    missed_count = int(np.sum(payment_array == 2))

    payment_records_available = len(payment_values)

    payment_issue_count = late_count + missed_count

    payment_issue_rate = (
        payment_issue_count / payment_records_available
    )

    missed_payment_rate = (
        missed_count / payment_records_available
    )

    on_time_rate = (
        on_time_count / payment_records_available
    )

    average_payment_status = float(
        np.mean(payment_array)
    )


    # --------------------------------------------------
    # 5. Prepare model input
    # --------------------------------------------------

    input_data = pd.DataFrame([{
        "Income": income,
        "Credit_Score": credit_score,
        "Credit_Utilization": credit_utilization,
        "Missed_Payments": missed_payments,
        "Loan_Balance": loan_balance,
        "Debt_to_Income_Ratio": debt_to_income,
        "Account_Tenure": account_tenure,
        "On_Time_Count": on_time_count,
        "Late_Count": late_count,
        "Missed_Count": missed_count,
        "Payment_Issue_Count": payment_issue_count,
        "Payment_Issue_Rate": payment_issue_rate,
        "Missed_Payment_Rate": missed_payment_rate,
        "On_Time_Rate": on_time_rate,
        "Average_Payment_Status": average_payment_status
    }])


    # --------------------------------------------------
    # 6. Predict risk
    # --------------------------------------------------

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    # Provisional thresholds used in this prototype
    if probability < 0.30:
        risk_category = "Low"
        intervention = (
            "Routine account monitoring and financial "
            "wellness reminders."
        )

    elif probability < 0.60:
        risk_category = "Medium"
        intervention = (
            "Friendly payment reminders and optional "
            "repayment guidance."
        )

    else:
        risk_category = "High"
        intervention = (
            "Priority human review and personalized "
            "repayment assistance."
        )


    # --------------------------------------------------
    # 7. Display results
    # --------------------------------------------------

    st.divider()
    st.header("Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predicted Delinquency Probability",
            f"{probability:.2%}"
        )

    with result_col2:
        st.metric(
            "Predicted Risk Category",
            risk_category
        )

    st.progress(probability)

    st.subheader("Recommended Support Intervention")
    st.write(intervention)

    st.subheader("Payment Behaviour Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.metric("On-time", on_time_count)
    summary_col2.metric("Late", late_count)
    summary_col3.metric("Missed", missed_count)

    st.subheader("Model Input Preview")
    st.dataframe(input_data, use_container_width=True)

    st.caption(
        "This is a model estimate, not a confirmed customer outcome. "
        "Human review is required before any consequential action."
    )
