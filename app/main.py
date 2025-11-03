import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------------------
# Load model and preprocessors
# ---------------------------
model = pickle.load(open(r'C:\Users\HP\Desktop\Innomatics\Machine Learning\Banking Domain\model\model.pkl', 'rb'))
scaler = pickle.load(open(r'C:\Users\HP\Desktop\Innomatics\Machine Learning\Banking Domain\model\scaler.pkl', 'rb'))
encoder = pickle.load(open(r'C:\Users\HP\Desktop\Innomatics\Machine Learning\Banking Domain\model\encoder.pkl', 'rb'))

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Credit Risk Scoring App", page_icon="💳", layout="wide")
st.title("💳 Credit Risk Scoring App")
st.write("Predict whether a loan applicant is **High Risk (Default)** or **Low Risk (Safe)**.")

st.sidebar.header("Enter Applicant Details")

# Numeric inputs
person_age = st.sidebar.number_input("Applicant Age", min_value=18, max_value=100, value=30)
person_income = st.sidebar.number_input("Annual Income ($)", min_value=0, step=1000, value=50000)
person_emp_length = st.sidebar.number_input("Employment Length (months)", min_value=0, max_value=500, value=12)
loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=500, max_value=50000, step=500, value=10000)
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, step=0.1, value=12.0)
loan_percent_income = st.sidebar.number_input("Loan Percent of Income", min_value=0.0, max_value=1.0, step=0.01, value=0.2)
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)

# Categorical inputs
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_intent = st.sidebar.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
cb_person_default_on_file = st.sidebar.selectbox("Previous Default on File", ["Y", "N"])

# ---------------------------
# Prepare input data
# ---------------------------
input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_home_ownership': [person_home_ownership],
    'person_emp_length': [person_emp_length],
    'loan_intent': [loan_intent],
    'loan_grade': [loan_grade],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_default_on_file': [cb_person_default_on_file],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length]
})

st.write("### Applicant Input Summary")
st.dataframe(input_data)

# ---------------------------
# Preprocessing
# ---------------------------
# Separate numeric and categorical columns
num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Scale and encode
num_scaled = scaler.transform(input_data[num_cols])
cat_encoded = encoder.transform(input_data[cat_cols])

# Combine
X_processed = np.concatenate([num_scaled, cat_encoded], axis=1)

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Predict Credit Risk"):
    prediction = model.predict(X_processed)
    prob = model.predict_proba(X_processed)[0][1] if hasattr(model, "predict_proba") else None

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk Applicant! (Default Probability: {prob:.2f})" if prob else "⚠️ High Risk Applicant!")
    else:
        st.success(f"✅ Low Risk Applicant (Safe to Approve). Default Probability: {prob:.2f}" if prob else "✅ Low Risk Applicant (Safe to Approve).")
