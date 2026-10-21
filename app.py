import streamlit as st
import pandas as pd
import joblib

# load train model

model = joblib.load(r"models\logistic_baseline.pkl")

expected_features = list(model.feature_names_in_)

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction")
st.write("Predict the probability that a customer will Churn")


# input form
with st.form("churn_form"):
    st.subheader("Customer Details")

    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total _Charges", min_value=0.0, value=840.0)

    contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
    key="contract"
    )

    internet_service = st.selectbox(
      "Internet Service",
      ["Fiber optic", "DSL", "No"]  
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Credit card (automatic)"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )

    submitted = st.form_submit_button("Predict Churn Risk")

#build Input Data

if submitted:
    input_data = dict.fromkeys(expected_features, 0)

    #numeirc features
    input_data["tenure"] = tenure
    input_data["monthly_charges"] = monthly_charges
    input_data["total_charges"] = total_charges
    input_data["early_tenure"] = 1 if tenure <= 12 else 0

    #Default binary assumption
    input_data["gender"] = 0
    input_data["SeniorCitizen"] = 0
    input_data["Partner"] = 0
    input_data["Dependents"] = 0
    input_data["PhoneService"] = 0

    #PaperLess Billing
    input_data["PaperlessBilling"] = 1 if paperless_billing == "Yes" else 0


    # contract billing
    if contract == "One year":
        input_data["Contract_One year"] = 1
    elif contract == "Two year":
        input_data["Contract_Two year"] = 1

    #Internet Service encoding
    if internet_service == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1
    elif internet_service == "No":
        input_data["InternetService_No"] = 1

    # Payment method encoding
    if payment_method == "Electronic check":
        input_data["PaymentMethod_Electronic check"] = 1
    elif payment_method == "Mailed check":
        input_data["PaymentMethod_Mailed check"] = 1
    elif payment_method == "Credit card (automatic)":
        input_data["PaymentMethod_Credit card (automatic)"] = 1

     # Services
    input_data["TechSupport_Yes"] = 1 if tech_support == "Yes" else 0

    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Reorder columns to match expected features
    input_df = input_df[expected_features]


    # Safety check
    assert list(input_df.columns) == expected_features

    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{prob:.2%}")

    if prob < 0.35:
        st.success("Low Risk of Churn")
    elif prob < 0.60:
        st.warning("Moderate Risk - monitor this customer")
    else:
        st.error("High Risk of Churn - immidiate retention action recommended")

with st.expander("ℹ️ How to interpret this result"):
    st.write(
        """
        - The model outputs a **probability of churn**, not a hard yes/no.
        - A threshold of **0.35** is used to flag high-risk customers.
        - This threshold was chosen to **maximize recall**, reducing missed churners.
        """
    )

st.markdown("---")
st.caption("Model: Logistic Regression | ROC-AUC: 0.86 | Business-focused churn prediction")
st.caption(
    "Try changing contract type or tenure to see how churn risk responds "
    "(scenario testing)."
)
