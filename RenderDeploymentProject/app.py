"""
Flask app: Medical Insurance Cost Predictor
---------------------------------------------
A minimal end-to-end deployment: loads a pre-trained Linear Regression model
and serves it two ways:
  1. A simple HTML form at "/" for humans to use in a browser
  2. A JSON API endpoint at "/predict" for programmatic access
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model + the exact feature column order it was trained on.
# We load this once at startup (not per-request) so predictions are fast.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "insurance_model.pkl")
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
feature_columns = model_bundle["feature_columns"]


def build_feature_row(age, sex, bmi, children, smoker, region):
    """
    Converts raw form inputs into a single-row DataFrame that exactly matches
    the feature columns/order the model was trained on (including one-hot
    encoded region columns).
    """
    row = {
        "age": age,
        "sex": 1 if sex == "male" else 0,
        "bmi": bmi,
        "children": children,
        "smoker": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }
    # Reindex to guarantee the exact same column order the model expects
    return pd.DataFrame([row])[feature_columns]


@app.route("/", methods=["GET"])
def home():
    """Renders the HTML form for browser-based use."""
    return render_template("index.html", prediction=None)


@app.route("/", methods=["POST"])
def predict_form():
    """Handles the HTML form submission and renders the result back on the page."""
    try:
        age = int(request.form["age"])
        sex = request.form["sex"]
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = request.form["smoker"]
        region = request.form["region"]

        features = build_feature_row(age, sex, bmi, children, smoker, region)
        prediction = model.predict(features)[0]
        prediction = round(float(prediction), 2)

        return render_template("index.html", prediction=prediction, error=None)
    except Exception as e:
        return render_template("index.html", prediction=None, error=str(e))


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    JSON API endpoint. Example request body:
    {
        "age": 29, "sex": "female", "bmi": 27.5,
        "children": 1, "smoker": "no", "region": "southeast"
    }
    """
    try:
        data = request.get_json(force=True)
        features = build_feature_row(
            age=int(data["age"]),
            sex=data["sex"],
            bmi=float(data["bmi"]),
            children=int(data["children"]),
            smoker=data["smoker"],
            region=data["region"],
        )
        prediction = model.predict(features)[0]
        return jsonify({"predicted_charges": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint — useful for Render's health monitoring."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # For local testing only. Render uses gunicorn (see Procfile) in production.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
