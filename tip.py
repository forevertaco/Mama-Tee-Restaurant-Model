import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("tip_predictor_model.pkl")
columns = joblib.load("model_columns.pkl")

st.title("Tip Prediction App - Mama Tee ML")
st.write("Enter customer information to predict tip")

# UI inputs
total_bill = st.number_input("Total Bill", min_value=0.0)
size = st.number_input("Size of Group", min_value=1)

gender = st.selectbox("Gender", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])

# Prepare input dict
input_data = {
    "total_bill": [total_bill],
    "size": [size],
    f"gender_{gender}": [1],
    f"smoker_{smoker}": [1],
    f"day_{day}": [1],
    f"time_{time}": [1]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# Align with model columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Predict
if st.button("Predict Tip"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Tip: ₦{prediction[0]:.2f}")
