
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("[INFO] Starting Model Evaluation...")

# 1. Load Dataset
df = pd.read_csv("data/StudentsPerformance.csv")

# 2. Create Target
def categorize(score):
    if score < 60:
        return "Low"
    elif score < 80:
        return "Good"
    else:
        return "Excellent"

df["Performance"] = df["writing score"].apply(categorize)

X = df.drop(columns=["writing score", "Performance"])
y = df["Performance"]

# 3. Recreate Same Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Load Trained Model
model = joblib.load("models/best_model.pkl")

# 5. Predict
y_pred = model.predict(X_test)

# 6. Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

print("\n========== EVALUATION RESULTS ==========")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Good", "Excellent"]
)

cm_df = pd.DataFrame(
    cm,
    index=["Low", "Good", "Excellent"],
    columns=["Low", "Good", "Excellent"]
)

# 8. Save Results
os.makedirs("output", exist_ok=True)

cm_df.to_csv("output/confusion_matrix.csv")

with open("output/evaluation_report.json", "w") as f:
    json.dump({
        "accuracy": accuracy,
        "classification_report": report
    }, f, indent=4)

print("\n[SUCCESS] Evaluation completed!")
print("[SUCCESS] Confusion matrix saved.")
print("[SUCCESS] Evaluation report saved.")
