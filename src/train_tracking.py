
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "StudentsPerformance.csv")

# --------------------------------------------------
# 2. MLFLOW CONFIGURATION
# --------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Student_Performance_Classification")


# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

print("\nLoading Student Performance Dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)


# --------------------------------------------------
# 4. CREATE TARGET VARIABLE
# --------------------------------------------------

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


# --------------------------------------------------
# 5. FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(
    columns=["writing score", "Performance"]
)

y = df["Performance"]


# --------------------------------------------------
# 6. IDENTIFY COLUMN TYPES
# --------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# --------------------------------------------------
# 7. PREPROCESSING
# --------------------------------------------------

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# --------------------------------------------------
# 8. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 9. DEFINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "SVM": SVC()
}


# --------------------------------------------------
# 10. MLFLOW EXPERIMENT TRACKING
# --------------------------------------------------

results = []

print("\nStarting MLflow Experiment Tracking...\n")

for model_name, model in models.items():

    print(f"Training {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    with mlflow.start_run(run_name=model_name):

        # Train model
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(
        pipeline,
        name="model",
        serialization_format="pickle"
        )

        # Store results
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        })

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 50)


# --------------------------------------------------
# 11. SAVE COMPARISON RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_dir = os.path.join(BASE_DIR, "output")

os.makedirs(output_dir, exist_ok=True)

results_path = os.path.join(
    output_dir,
    "mlflow_model_comparison.csv"
)

results_df.to_csv(results_path, index=False)

print("\nMLflow Experiment Tracking Completed!")

print("\nModel Comparison:")
print(results_df.to_string(index=False))

print("\nResults saved at:", results_path)

print("\nExperiment Name: Student_Performance_Classification")
print("Tracking URI: sqlite:///mlflow.db")
