import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib, os

df = pd.read_csv('data/heart_uci_combined.csv')

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('clf', XGBClassifier(random_state=42, eval_metric='logloss'))
])

param_dist = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5],
    'clf__learning_rate': [0.01, 0.1],
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=5,
    scoring='roc_auc',
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=2
)

search.fit(X, y)

print("Best ROC-AUC:", search.best_score_)

os.makedirs('models', exist_ok=True)
joblib.dump(search.best_estimator_, os.path.join('models', 'best_model_combined.joblib'))



