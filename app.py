# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# === Load trained model and artifacts ===
with open("model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
encoders = artifacts["encoders"]
feature_columns = artifacts["feature_columns"]

# === Load dataset to get all locations dynamically ===
df = pd.read_csv("credit_card_fraud_dataset.csv")
locations = sorted(df["Location"].unique())

# === ROUTES ===

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/input")
def input_page():
    return render_template("input.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Collect form data
    form = request.form
    input_data = {}

    for col in feature_columns:
        val = form.get(col, 0)  # default to 0 if not provided
        input_data[col] = val

    df_row = pd.DataFrame([input_data], columns=feature_columns)

    # 2. Encode categorical columns
    for col, le in encoders.items():
        raw_val = str(df_row.loc[0, col])
        if raw_val not in le.classes_:
            return render_template(
                "output.html",
                error=f"Invalid value for {col}: '{raw_val}'. Allowed: {list(le.classes_)}"
            )
        df_row[col] = le.transform([raw_val])[0]

    # 3. Convert numeric columns
    for c in df_row.columns:
        try:
            df_row[c] = pd.to_numeric(df_row[c])
        except:
            df_row[c] = 0

    # 4. Scale features
    X_scaled = scaler.transform(df_row.values)

    # 5. Predict
    pred = model.predict(X_scaled)[0]
    label = "Fraud" if pred == 0 else "Not Fraud"

    # 6. Render output
    return render_template(
        "output.html",
        prediction=int(pred),
        label=label,
        input_row=df_row.to_dict(orient="records")[0]
    )

# === RUN APP ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



