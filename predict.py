import joblib

# Load model
model = joblib.load("heart_model.pkl")

# Take input from user
print("Enter patient data:")
age = int(input("Age: "))
sex = int(input("Sex (1 = male, 0 = female): "))
cp = int(input("Chest Pain Type (0 = typical, 1 = atypical, 2 = non-anginal, 3 = asymptomatic): "))
trestbps = int(input("Resting Blood Pressure (e.g., 120): "))
chol = int(input("Cholesterol (e.g., 200): "))

import pandas as pd

features = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol
}])


# Predict
prediction = model.predict(features)
proba = model.predict_proba(features)

# Output
print("\nPrediction (1 = Heart Disease, 0 = No Heart Disease):", prediction[0])
print("Confidence:", proba[0])
