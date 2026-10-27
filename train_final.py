from pathlib import Path
import pandas as pd
import joblib as jb
from sklearn.ensemble import RandomForestClassifier

FEATURES = ['bmi', 'hba1c_level', 'age', 'fasting_blood_sugar', 'waist_circumference_cm']
data = pd.read_csv(Path(__file__).resolve().parent/"diabetes_risk.csv")
X = data[FEATURES]
y = (data['diabetes_risk'] == 'High').astype(int)

model = RandomForestClassifier(max_depth=5, random_state=42)
model.fit(X, y)
jb.dump({'model': model, 'features': FEATURES, 'threshold': 0.2}, Path(__file__).resolve().parent/"risk_model.pkl")

#load the model and make predictions
saved_model = jb.load('risk_model.pkl')
patient_data = pd.DataFrame([{'bmi': 29.0, 'hba1c_level': 7.5, 'age': 52, 'fasting_blood_sugar': 130, 'waist_circumference_cm': 102}])
p = saved_model["model"].predict_proba(patient_data[saved_model["features"]])[0, 1]
print("probability:", round(p, 3), "| flagged:", bool(p >= saved_model["threshold"]))

