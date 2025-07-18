import joblib
import numpy as np
import pandas as pd

def get_user_input():
    print("\n📊 Please enter the following health information:")
    age = float(input("Age: "))
    sex = int(input("Sex (0 = female, 1 = male): "))
    cp = int(input("Chest Pain Type (0–3): "))
    resting_bp = float(input("Resting Blood Pressure: "))
    chol = float(input("Cholesterol: "))
    fasting_bs = float(input("Fasting Blood Sugar (0 or 1): "))
    rest_ecg = int(input("Resting ECG (0–2): "))
    max_hr = float(input("Max Heart Rate: "))
    ex_angina = int(input("Exercise Induced Angina (0 = no, 1 = yes): "))
    oldpeak = float(input("Oldpeak (ST depression): "))
    st_slope = int(input("ST Slope (0–2): "))

    return pd.DataFrame([[age, sex, cp, resting_bp, chol, fasting_bs, rest_ecg,
                          max_hr, ex_angina, oldpeak, st_slope]],
                        columns=[
                            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                            'Oldpeak', 'ST_Slope'
                        ])

if __name__ == "__main__":
    model = joblib.load('models/best_model_combined.joblib')
    user_df = get_user_input()
    prediction = model.predict_proba(user_df)[0][1]
    print(f"\n🫀 Estimated Heart Disease Risk: {prediction:.2%}")
    
if prediction < 0.2:
    print("🟢 Low Risk")
elif prediction < 0.6:
    print("🟡 Moderate Risk")
else:
    print("🔴 High Risk")


