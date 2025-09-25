# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# === Load artifacts safely ===
with open("model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts.get("model")
scaler = artifacts.get("scaler")
encoders = artifacts.get("encoders")
feature_columns = artifacts.get("feature_columns")

best_model_name = artifacts.get("best_model_name", "Unknown Model")
best_accuracy = artifacts.get("best_accuracy", 0.0)
best_params = artifacts.get("best_params", {})

print(f"✅ Loaded best model: {best_model_name}")
print(f"📊 Accuracy: {best_accuracy:.4f}")
print(f"⚙️ Best Hyperparameters: {best_params}")


@app.route("/")
def home():
    return render_template("welcome.html")


@app.route("/input")
def input_page():
    locations = encoders["Location"].classes_ if "Location" in encoders else []
    transaction_types = encoders["TransactionType"].classes_ if "TransactionType" in encoders else []
    return render_template("input.html", locations=locations, transaction_types=transaction_types)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # === Collect and validate form data ===
        try:
            transaction_id = int(request.form["TransactionID"])
        except ValueError:
            return render_template("error.html", error_message="Transaction ID must be an integer.")

        try:
            amount = float(request.form["Amount"])
        except ValueError:
            return render_template("error.html", error_message="Amount must be a number.")

        try:
            merchant_id = int(request.form["MerchantID"])
        except ValueError:
            return render_template("error.html", error_message="Merchant ID must be an integer.")

        transaction_type = request.form["TransactionType"]
        location = request.form["Location"]

        # === Encode categorical inputs safely ===
        if transaction_type not in encoders["TransactionType"].classes_:
            return render_template("error.html", error_message="Invalid Transaction Type.")
        if location not in encoders["Location"].classes_:
            return render_template("error.html", error_message="Invalid Location.")

        transaction_type_encoded = encoders["TransactionType"].transform([transaction_type])[0]
        location_encoded = encoders["Location"].transform([location])[0]

        # === Build feature vector in correct order ===
        input_dict = {
            "TransactionID": transaction_id,
            "Amount": amount,
            "MerchantID": merchant_id,
            "TransactionType": transaction_type_encoded,
            "Location": location_encoded
        }

        input_array = np.array([input_dict[col] for col in feature_columns]).reshape(1, -1)

        # === Scale numerical features ===
        input_scaled = scaler.transform(input_array)

        # === Predict ===
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  # probability of fraud

        # === Format result ===
        result = "❌ Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        result += f" (Confidence: {probability:.2%})"

        return render_template("output.html", prediction=result)

    except Exception as e:
        return render_template("error.html", error_message=f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
