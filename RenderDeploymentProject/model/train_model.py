"""
Trains the Medical Insurance Cost Predictor model and saves it as insurance_model.pkl.
Run this once locally to regenerate the model file if needed:
    python model/train_model.py
"""

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load the same public dataset used in Assignment 1
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

# Encode categorical variables
df_encoded = df.copy()
df_encoded["sex"] = df_encoded["sex"].map({"male": 1, "female": 0})
df_encoded["smoker"] = df_encoded["smoker"].map({"yes": 1, "no": 0})
df_encoded = pd.get_dummies(df_encoded, columns=["region"], drop_first=True)

feature_columns = [c for c in df_encoded.columns if c != "charges"]

X = df_encoded[feature_columns]
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# Save the model together with its expected feature column order, so app.py can
# reliably rebuild the correct input format at inference time.
joblib.dump({"model": model, "feature_columns": feature_columns}, "model/insurance_model.pkl")
print("Model saved to model/insurance_model.pkl")
