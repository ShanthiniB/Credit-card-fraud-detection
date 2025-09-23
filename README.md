This project is a Credit Card Fraud Detection System that uses machine learning (Logistic Regression) to predict whether a given transaction is fraudulent or not.

The dataset contains transaction details like Transaction ID, Amount, Merchant ID, Transaction Type, and Location.

Categorical values (e.g., Transaction Type, Location) are label-encoded for model training.

Features are standardized using StandardScaler to improve model accuracy.

A Logistic Regression model (with class balancing for fraud cases) is trained and saved as model.pkl.

A Flask web app is built to allow users to enter transaction details and get real-time predictions:

0 → Fraudulent Transaction (❌ Fraud)

1 → Legitimate Transaction (✅ Not Fraud)

This project demonstrates how machine learning + web deployment can be used to detect fraudulent credit card activity effectively.
