
import os
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd

from mlflow import MlflowClient
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

DATA_PATH = os.path.join(BASE_DIR, "data", "StudentsPerformance.csv")


# --------------------------------------------------
# 2. MLFLOW CONFIGURATION
# --------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

EXPERIMENT_NAME = "Student_Performance_Classification"

REGISTERED_MODEL_NAME = "Student_Performance_Model"

mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

print("=" * 60)
print("LAB 6: MLFLOW MODEL REGISTRY")
print("=" * 60)

print("\n[INFO] Loading trained model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "best_model.pkl not found. Run Lab 3 first."
    )

model = joblib.load(MODEL_PATH)

print("[SUCCESS] Model loaded successfully.")


# --------------------------------------------------
# 4. LOAD DATA FOR VALIDATION
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


def categorize_performance(score):

    if score < 60:
        return "Low"

    elif score < 80:
        return "Good"

    else:
        return "Excellent"


df["Performance"] = df["writing score"].apply(
    categorize_performance
)

X = df.drop(
    columns=["writing score", "Performance"]
)

y = df["Performance"]


# --------------------------------------------------
# 5. MODEL VALIDATION
# --------------------------------------------------

print("\n[INFO] Validating model...")

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)

print(f"Validation Accuracy: {accuracy:.4f}")


# --------------------------------------------------
# 6. REGISTER MODEL IN MLFLOW
# --------------------------------------------------

print("\n[INFO] Registering model...")

with mlflow.start_run(run_name="Model_Registry") as run:

    mlflow.log_param("model_name", "Best Student Performance Model")

    mlflow.log_metric("validation_accuracy", accuracy)

    mlflow.sklearn.log_model(
        model,
        name="registered_model",
        serialization_format="pickle",
        registered_model_name=REGISTERED_MODEL_NAME
    )

    run_id = run.info.run_id

print("[SUCCESS] Model registration completed.")


# --------------------------------------------------
# 7. GET LATEST MODEL VERSION
# --------------------------------------------------

print("\n[INFO] Retrieving registered model version...")

versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)

if not versions:
    raise RuntimeError("Model registration failed.")

latest_version = max(
    versions,
    key=lambda version: int(version.version)
)

model_version = latest_version.version

print("Registered Model:", REGISTERED_MODEL_NAME)
print("Model Version:", model_version)


# --------------------------------------------------
# 8. ASSIGN PRODUCTION ALIAS
# --------------------------------------------------

print("\n[INFO] Assigning production alias...")

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="production",
    version=model_version
)

print("[SUCCESS] Production alias assigned.")


# --------------------------------------------------
# 9. LOAD REGISTERED MODEL
# --------------------------------------------------

print("\n[INFO] Loading registered model...")

model_uri = f"models:/{REGISTERED_MODEL_NAME}@production"

registered_model = mlflow.sklearn.load_model(model_uri)

print("[SUCCESS] Registered model loaded.")


# --------------------------------------------------
# 10. SAMPLE PREDICTION
# --------------------------------------------------

sample = X.iloc[[0]]

prediction = registered_model.predict(sample)

print("\nSample Prediction:", prediction[0])


# --------------------------------------------------
# 11. FINAL STATUS
# --------------------------------------------------

print("\n" + "=" * 60)
print("LAB 6 MODEL REGISTRY COMPLETED!")
print("=" * 60)

print("Model Name:", REGISTERED_MODEL_NAME)
print("Model Version:", model_version)
print("Model Alias: production")
print("Validation Accuracy:", round(accuracy, 4))
