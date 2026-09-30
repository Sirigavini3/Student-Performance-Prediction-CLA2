
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "StudentsPerformance.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("LAB 5: PRODUCTION DATA PIPELINE")
print("=" * 60)

print("\n[INFO] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("[INFO] Dataset Shape:", df.shape)


# --------------------------------------------------
# 3. SCHEMA VALIDATION
# --------------------------------------------------

print("\n[INFO] Validating dataset schema...")

required_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "math score",
    "reading score",
    "writing score"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df.empty:
    raise ValueError("Dataset is empty!")

print("[SUCCESS] Schema validation passed.")


# --------------------------------------------------
# 4. DATA QUALITY VALIDATION
# --------------------------------------------------

print("\n[INFO] Checking data quality...")

print("Missing values before preprocessing:")
print(df.isnull().sum())

print("\nDuplicate records:", df.duplicated().sum())

for col in ["math score", "reading score", "writing score"]:

    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"{col} must be numeric.")

    if not df[col].between(0, 100).all():
        raise ValueError(f"Invalid score values in {col}.")

print("[SUCCESS] Data quality validation passed.")


# --------------------------------------------------
# 5. CREATE TARGET VARIABLE
# --------------------------------------------------

print("\n[INFO] Creating performance labels...")

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

print(df["Performance"].value_counts())


# --------------------------------------------------
# 6. FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(
    columns=["writing score", "Performance"]
)

y = df["Performance"]


# --------------------------------------------------
# 7. IDENTIFY FEATURE TYPES
# --------------------------------------------------

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["str", "object", "category"]
).columns.tolist()

print("\nNumerical Features:", numerical_features)
print("Categorical Features:", categorical_features)


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

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 9. BUILD PREPROCESSING PIPELINE
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
# 10. FIT PREPROCESSOR
# --------------------------------------------------

print("\n[INFO] Fitting preprocessing pipeline...")

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("[SUCCESS] Preprocessing completed.")

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)


# --------------------------------------------------
# 11. SAVE PREPROCESSOR
# --------------------------------------------------

preprocessor_path = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

joblib.dump(preprocessor, preprocessor_path)

print("\n[SUCCESS] Preprocessor saved:", preprocessor_path)


# --------------------------------------------------
# 12. SAVE PROCESSED DATA
# --------------------------------------------------

train_path = os.path.join(
    OUTPUT_DIR,
    "X_train_processed.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "X_test_processed.csv"
)

pd.DataFrame(
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed
).to_csv(train_path, index=False)

pd.DataFrame(
    X_test_processed.toarray()
    if hasattr(X_test_processed, "toarray")
    else X_test_processed
).to_csv(test_path, index=False)

print("[SUCCESS] Processed datasets saved.")


# --------------------------------------------------
# 13. FINAL VALIDATION
# --------------------------------------------------

loaded_preprocessor = joblib.load(preprocessor_path)

sample_output = loaded_preprocessor.transform(
    X_test.iloc[:5]
)

if sample_output.shape[0] != 5:
    raise ValueError("Output validation failed!")

print("\n[SUCCESS] Saved preprocessor loaded successfully.")
print("[SUCCESS] Output validation passed.")

print("\n" + "=" * 60)
print("LAB 5 PRODUCTION PIPELINE COMPLETED!")
print("=" * 60)
