import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/heart_edited.csv")

from sklearn.preprocessing import LabelEncoder
if df['RestingECG'].dtype == 'object':
    le = LabelEncoder()
    df['RestingECG'] = le.fit_transform(df['RestingECG'])

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = joblib.load("models/best_model.joblib")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC Score: {:.4f}".format(roc_auc_score(y_test, y_proba)))
