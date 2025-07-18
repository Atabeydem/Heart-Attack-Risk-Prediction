import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

df = pd.read_csv(os.path.join("data", "heart_uci_combined.csv"))
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(rf, os.path.join("models", "final_model.joblib"))
joblib.dump(scaler, os.path.join("models", "scaler.joblib"))

print("Final model and scaler saved successfully.")


