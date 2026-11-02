
import os
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/StudentsPerformance.csv"
OUTPUT_DIR = "output"

ACCURACY_THRESHOLD = 0.80
F1_THRESHOLD = 0.80

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Create target variable
def classify_performance(score):
    if score < 60:
        return "Low"
    elif score < 80:
        return "Good"
    else:
        return "Excellent"

df["Performance"] = df["writing score"].apply(
    classify_performance
)

# Prepare features and target
X = df.drop(columns=["writing score", "Performance"])
y = df["Performance"]

# -----------------------------
# Load Trained Model
# -----------------------------

print("Loading trained model...")

model = joblib.load(MODEL_PATH)

# -----------------------------
# Model Prediction
# -----------------------------

print("Generating predictions...")

y_pred = model.predict(X)

# -----------------------------
# Calculate Metrics
# -----------------------------

accuracy = accuracy_score(y, y_pred)

precision = precision_score(
    y, y_pred, average="weighted", zero_division=0
)

recall = recall_score(
    y, y_pred, average="weighted", zero_division=0
)

f1 = f1_score(
    y, y_pred, average="weighted", zero_division=0
)

report = classification_report(
    y, y_pred, output_dict=True, zero_division=0
)

# -----------------------------
# Quality Gate
# -----------------------------

quality_status = (
    "PASS"
    if accuracy >= ACCURACY_THRESHOLD
    and f1 >= F1_THRESHOLD
    else "FAIL"
)

# -----------------------------
# Save Results
# -----------------------------

results = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "accuracy_threshold": ACCURACY_THRESHOLD,
    "f1_threshold": F1_THRESHOLD,
    "quality_status": quality_status,
    "classification_report": report
}

report_path = os.path.join(
    OUTPUT_DIR, "lab7_quality_report.json"
)

with open(report_path, "w") as file:
    json.dump(results, file, indent=4)

# -----------------------------
# Display Results
# -----------------------------

print("\n========== LAB 7 RESULTS ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nQuality Gate:", quality_status)

print("\nReport saved at:", report_path)

if quality_status == "PASS":
    print("Model passed the quality checks.")
else:
    print("Model failed the quality checks.")
