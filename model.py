# model.py
import numpy as np
import pandas as pd
import pickle
import os
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
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
        return
    
    try:
        credit = pd.read_csv(CSV_PATH)
        print("✅ Dataset loaded with shape:", credit.shape)
        print("Dataset columns:", list(credit.columns))
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # Check for required columns
    required_columns = ["IsFraud"]
    missing_cols = [col for col in required_columns if col not in credit.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return

    # 2. Encode categorical features (only if they exist)
    encoders = {}
    categorical_cols = ["TransactionType", "Location"]
    for col in categorical_cols:
        if col in credit.columns:
            le = LabelEncoder()
            credit[col] = le.fit_transform(credit[col].astype(str))
            encoders[col] = le
            print(f"✅ Encoded column: {col}")
        else:
            print(f"⚠️ Warning: Column '{col}' not found in dataset")

    # 3. Features and target
    # Drop non-feature columns
    columns_to_drop = ["TransactionDate", "IsFraud"]
    existing_drop_cols = [col for col in columns_to_drop if col in credit.columns]
    X = credit.drop(existing_drop_cols, axis=1)

    # Check if target column exists
    if "IsFraud" not in credit.columns:
        print("❌ Error: Target column 'IsFraud' not found")
        return

    # Flip labels → 1 = Not Fraud, 0 = Fraud
    y = 1 - credit["IsFraud"]

    feature_columns = list(X.columns)
    print("🔎 Feature columns:", feature_columns)
    print("✅ Target distribution after flipping:")
    print(y.value_counts())

    # Validate we have features
    if X.empty or len(feature_columns) == 0:
        print("❌ Error: No feature columns found")
        return

    # 4. Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"✅ Train set size: {x_train.shape}, Test set size: {x_test.shape}")

    # 5. Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 6. Simplified model selection (reduced for faster training)
    param_grids = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["lbfgs"]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 150],
                "learning_rate": [0.1, 0.2],
                "max_depth": [3, 5]
            }
        }
    }

    best_model = None
    best_acc = 0
    best_name = ""
    best_params = {}

    # 7. Train & evaluate each model
    for name, cfg in param_grids.items():
        print(f"\n🔍 Tuning {name}...")
        try:
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
            print(classification_report(y_test, preds, target_names=['Fraud', 'Not Fraud']))

            if acc > best_acc:
                best_acc = acc
                best_model = best
                best_name = name
                best_params = grid.best_params_
                
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            continue

    if best_model is None:
        print("❌ Error: No model was successfully trained")
        return

    print(f"\n🏆 Best Model: {best_name} with accuracy {best_acc:.4f}")
    print(f"Best hyperparameters: {best_params}")

    # 8. Save artifacts
    artifacts = {
        "model": best_model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_columns": feature_columns,
        "best_model_name": best_name,
        "best_accuracy": best_acc,
        "best_params": best_params
    }

    try:
        with open(OUTPUT_PICKLE, "wb") as f:
            pickle.dump(artifacts, f)
        print(f"✅ Best model saved to {OUTPUT_PICKLE}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return

    # 9. Test with sample data (if we have enough features)
    if len(feature_columns) >= 5:
        # Create sample input with correct number of features
        sample_values = [191, 1703.8, 916, 0, 9]
        # Pad or truncate to match actual feature count
        if len(feature_columns) > len(sample_values):
            sample_values.extend([0] * (len(feature_columns) - len(sample_values)))
        elif len(feature_columns) < len(sample_values):
            sample_values = sample_values[:len(feature_columns)]
            
        input_array = np.array(sample_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = best_model.predict(input_scaled)[0]
        print(f"🔹 Test input prediction: {'✅ Not Fraud' if prediction == 1 else '❌ Fraud'}")

if __name__ == "__main__":
    train_and_save()