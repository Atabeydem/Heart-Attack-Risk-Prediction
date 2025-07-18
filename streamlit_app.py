import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/best_model_combined.joblib")

st.markdown("## 💗 Heart Attack Risk Prediction App")

st.markdown("This app estimates your risk of heart attack based on some basic health data.")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[(0, "Female"), (1, "Male")], format_func=lambda x: x[1])[0]
cp = st.selectbox("Chest Pain Type", options=[(0, "Typical Angina"), (1, "Atypical Angina"), (2, "Non-anginal Pain"), (3, "Asymptomatic")], format_func=lambda x: x[1])[0]
resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
rest_ecg = st.selectbox("Resting ECG", options=[(0, "Normal"), (1, "ST-T wave abnormality"), (2, "Left ventricular hypertrophy")], format_func=lambda x: x[1])[0]
max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
ex_angina = st.selectbox("Exercise Induced Angina", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox("ST Slope", options=[(0, "Up"), (1, "Flat"), (2, "Down")], format_func=lambda x: x[1])[0]

if st.button("Predict"):
    user_input = pd.DataFrame([[age, sex, cp, resting_bp, chol, fasting_bs,
                                rest_ecg, max_hr, ex_angina, oldpeak, st_slope]],
                              columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                                       'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                                       'Oldpeak', 'ST_Slope'])

    risk = model.predict_proba(user_input)[0][1]
    st.write(f"### 🫀 Estimated Heart Disease Risk: **{risk:.2%}**")

    if risk < 0.3:
        st.success("✅ Low Risk. Keep monitoring your health.")
    elif risk < 0.7:
        st.warning("⚠️ Medium Risk. Consider a health check-up.")
    else:
        st.error("🚨 High Risk! Please consult a doctor immediately.")

