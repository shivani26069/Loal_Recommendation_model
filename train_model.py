import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset (ensure this file is in the same folder)
df = pd.read_excel("final_loan_data EXCEL.xlsx")

# Create a simple approval rule for training
df['Loan_Approved'] = (
    (df['CIBIL_Score'] > 700) &
    (df['EMI_Bounces'] < 2) &
    (df['EMI_Amount'] < df['Salary'] * 0.4) &
    (df['Existing_Loan_Amount'] < df['Salary'] * 6)
).astype(int)

# Encode "Has_Existing_Loan"
df['Has_Existing_Loan'] = df['Has_Existing_Loan'].map({'Yes': 1, 'No': 0})

# Define features and target
features = ['Age', 'Salary', 'CIBIL_Score', 'EMI_Amount', 'EMI_Bounces', 'Has_Existing_Loan', 'Existing_Loan_Amount']
X = df[features]
y = df['Loan_Approved']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "loan_model.pkl")

print("✅ Model trained and saved as loan_model.pkl")
