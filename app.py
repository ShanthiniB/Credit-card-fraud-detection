import pickle
import numpy as np
from flask import Flask, request, render_template

# Load artifacts
with open("model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
feature_columns = artifacts["feature_columns"]

print(f"✅ Loaded best model: {artifacts['best_model_name']} "
      f"(Accuracy: {artifacts['best_accuracy']:.4f})")

# Flask app
app = Flask(__name__, template_folder="templates")  # 👈 ensure correct template path

@app.route("/")
def home():
    return render_template("welcome.html")

@app.route("/input")
def input_page():   # 👈 add a route for input.html
    return render_template("input.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input features in correct order
        input_features = [float(request.form.get(col, 0)) for col in feature_columns]
        input_array = np.array(input_features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        # Interpret prediction (flipped labels as per your request)
        result = "✅ Not Fraud" if prediction == 1 else "❌ Fraud"

        return render_template("output.html",
                               prediction_text=f"Transaction Prediction: {result}")

    except Exception as e:
        return render_template("output.html",
                               prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
