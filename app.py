# app.py
import pickle
import numpy as np
import os
from flask import Flask, request, render_template

# Check if model file exists
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file '{MODEL_PATH}' not found. Please run model.py first.")
    exit(1)

try:
    # Load artifacts
    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_columns = artifacts["feature_columns"]
    encoders = artifacts.get("encoders", {})

    print(f"✅ Loaded best model: {artifacts['best_model_name']} "
          f"(Accuracy: {artifacts['best_accuracy']:.4f})")
    print(f"🔎 Feature columns: {feature_columns}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Create templates directory if it doesn't exist
if not os.path.exists("templates"):
    os.makedirs("templates")
    print("✅ Created templates directory")

# Flask app
app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("welcome.html")

@app.route("/input")
def input_page():
    return render_template("input.html", feature_columns=feature_columns)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input features in correct order
        input_features = []
        for col in feature_columns:
            value = request.form.get(col, 0)
            try:
                input_features.append(float(value))
            except ValueError:
                return render_template("output.html",
                                     prediction_text=f"❌ Error: Invalid value for {col}: '{value}'")
        
        # Validate input
        if len(input_features) != len(feature_columns):
            return render_template("output.html",
                                 prediction_text=f"❌ Error: Expected {len(feature_columns)} features, got {len(input_features)}")

        input_array = np.array(input_features).reshape(1, -1)
        
        # Scale input
        input_scaled = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # Interpret prediction (flipped labels)
        result = "✅ Not Fraud" if prediction == 1 else "❌ Fraud"
        confidence = max(prediction_proba) * 100

        return render_template("output.html",
                             prediction_text=f"Transaction Prediction: {result}",
                             confidence=f"Confidence: {confidence:.1f}%")

    except Exception as e:
        return render_template("output.html",
                             prediction_text=f"❌ Error: {str(e)}")

@app.errorhandler(404)
def not_found_error(error):
    return render_template("output.html",
                         prediction_text="❌ Error: Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("output.html",
                         prediction_text="❌ Error: Internal server error"), 500

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)