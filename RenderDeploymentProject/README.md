# End-to-End Model Deployment — Medical Insurance Cost Predictor (Flask + Render)

## 📌 Objective
Take a trained machine learning model all the way from training to a **live, publicly
accessible web app** — a full end-to-end deployment pipeline using Flask and Render.

This reuses the Linear Regression model from the Medical Insurance Cost Prediction assignment,
wraps it in a simple Flask web app, and deploys it to [Render](https://render.com) (a free
hosting platform) so anyone can use it from a browser or call it as an API.

## 🧠 What It Does
- A simple web form where you enter age, sex, BMI, number of children, smoker status, and
  region, and get back a predicted annual insurance cost.
- A JSON API endpoint (`/predict`) for programmatic access — e.g. from another app, a script,
  or a tool like Postman.
- A health check endpoint (`/health`) that Render uses to monitor the app is running.

## 🛠️ Tech Stack
- `Flask` — lightweight Python web framework
- `scikit-learn` — the trained Linear Regression model
- `gunicorn` — production-grade WSGI server (Render runs the app through this, not Flask's
  built-in dev server)
- `joblib` — model serialization/loading
- Plain HTML/CSS — no JavaScript framework needed for a form this simple

## 📂 Project Structure
```
RenderDeploymentProject/
├── app.py                      # Flask app: routes, model loading, prediction logic
├── requirements.txt            # Python dependencies
├── Procfile                    # Tells Render how to start the app (gunicorn)
├── model/
│   ├── insurance_model.pkl     # Pre-trained model (+ feature column order)
│   └── train_model.py          # Script to retrain the model from scratch
├── templates/
│   └── index.html              # HTML form (Jinja2 template)
└── static/
    └── style.css                # Styling for the form
```

## 🚀 How to Deploy on Render (Step by Step)

### 1. Push this folder to GitHub
Make sure this entire folder (with all files above) is committed to your GitHub repo.

### 2. Create a Render account
Go to [render.com](https://render.com) and sign up (free tier available, no credit card
required for a basic web service).

### 3. Create a new Web Service
- Click **New +** → **Web Service**
- Connect your GitHub account and select your repository
- If this project lives in a subfolder of a larger repo, set **Root Directory** to the path of
  this folder (e.g. `RenderDeploymentProject`)

### 4. Configure the service
- **Name:** anything you like (e.g. `insurance-cost-predictor`)
- **Environment:** Python 3
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app` (this is also what the `Procfile` specifies)
- **Instance Type:** Free

### 5. Deploy
Click **Create Web Service**. Render will install dependencies, then start your app. This
takes a few minutes on the first deploy. Once live, Render gives you a public URL like:
```
https://insurance-cost-predictor.onrender.com
```

### 6. Test it
- Visit the URL in your browser → you should see the form
- Or test the API directly:
```bash
curl -X POST https://your-app-name.onrender.com/predict \\
  -H "Content-Type: application/json" \\
  -d '{"age": 29, "sex": "female", "bmi": 27.5, "children": 1, "smoker": "no", "region": "southeast"}'
```

**Note:** Render's free tier spins down inactive services after a period of no traffic, so the
first request after idle time may take 30-60 seconds to "wake up" the app — this is normal
free-tier behavior, not a bug.

## 🧪 Running Locally First (Recommended Before Deploying)
```bash
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

## 📈 Model Details
- **Algorithm:** Linear Regression (scikit-learn)
- **Features:** age, sex, BMI, children, smoker status, region (one-hot encoded)
- **Performance:** MAE ≈ $4,181, R² ≈ 0.78 (same model/results as the original Assignment 1
  notebook — see `model/train_model.py` for the exact training code)

## ✅ Conclusion
This project demonstrates a complete, minimal MLOps pipeline: training a model, serializing it
for reuse, wrapping it in a REST API and simple web interface, and deploying it to a live,
publicly accessible platform. While production systems typically add more — authentication,
request logging, model versioning, monitoring/alerting, and CI/CD pipelines for automatic
redeployment on model updates — this project covers the core deployment loop that every ML
system eventually needs: getting a trained model out of a notebook and into something other
people (or other systems) can actually use.
