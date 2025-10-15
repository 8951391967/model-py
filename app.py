import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load  # if you saved your model
from model import linearregression
import joblib


# Load pre-trained model and scaler
model = load("logistic_model.joblib")   # your scratch model saved
scaler = load("scaler.joblib")

st.title("Diabetes Prediction")

# Get user inputs
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)

# Button to predict
if st.button("Predict"):
    # Convert to numpy array
    X_user = np.array([[pregnancies, glucose, blood_pressure,
                        skin_thickness, insulin, bmi, dpf, age]])
    
    # Handle zeros if necessary (same as training)
    X_user[:, [1,2,3,4,5]] = np.where(X_user[:, [1,2,3,4,5]]==0, 
                                      np.nan, X_user[:, [1,2,3,4,5]])
    # Fill nan with median values (you can hardcode medians from training set)
    medians = np.array([6, 120, 69, 20, 79, 32])  # example median values
    for i, col in enumerate([0,1,2,3,4,5]):
        if np.isnan(X_user[0, col]):
            X_user[0, col] = medians[i]

    # Scale features
    X_user_scaled = scaler.transform(X_user)

    # Predict
    pred = model.probict(X_user_scaled)
    prob = model.pridict_prob(X_user_scaled)[0,1]

    if pred[0] == 1:
        st.write(f"The model predicts **Diabetes** with probability {prob[0]:.2f}")
    else:
        st.write(f"The model predicts **No Diabetes** with probability {prob[0]:.2f}")
