# model.py
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# === CONFIG ===
CSV_PATH = r"C:/Users/Shanthini/OneDrive/Desktop/Project 2/credit_card_fraud_dataset.csv"
OUTPUT_PICKLE = "model.pkl"


def train_and_save():
    # 1. Load dataset
    credit = pd.read_csv(CSV_PATH)
    print("✅ Dataset loaded with shape:", credit.shape)

    # 2. Encode categorical features
    encoders = {}
    for col in ["TransactionType", "Location"]:
        le = LabelEncoder()
        credit[col] = le.fit_transform(credit[col].astype(str))
        encoders[col] = le

    # 3. Features and target
    X = credit.drop(["TransactionDate", "IsFraud"], axis=1, errors="ignore")
    y = credit["IsFraud"]

    feature_columns = list(X.columns)
    print("🔎 Feature columns:", feature_columns)

    # 4. Train-test split (no SMOTE, just raw data)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 5. Scale features
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 6. Candidate models with hyperparameters
    param_grids = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        }
    }

    best_model = None
    best_acc = 0
    best_name = ""

    # 7. Train & evaluate each model
    for name, cfg in param_grids.items():
        print(f"\n🔍 Tuning {name}...")
        grid = GridSearchCV(
            estimator=cfg["model"],
            param_grid=cfg["params"],
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(x_train_scaled, y_train)
        best = grid.best_estimator_

        preds = best.predict(x_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"📊 {name} Best Params: {grid.best_params_}")
        print(f"📊 {name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

        if acc > best_acc:
            best_acc = acc
            best_model = best
            best_name = name

    print(f"\n🏆 Best Model: {best_name} with accuracy {best_acc:.4f}")

    # 8. Save artifacts
    artifacts = {
        "model": best_model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_columns": feature_columns,
        "best_model_name": best_name,
        "best_accuracy": best_acc
    }

    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"✅ Best model saved to {OUTPUT_PICKLE}")


if __name__ == "__main__":
    train_and_save()
