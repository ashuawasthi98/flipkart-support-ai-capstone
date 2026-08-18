import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# =====================================================================
# TASK 1 & 2: Load the verified dataset
# =====================================================================
df = pd.read_csv("orders_dataset.csv")

# Separate features (X) and target (y). Drop identifier column 'order_id'.
X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

# =====================================================================
# TASK 3: Stratified 80/20 train/test split (leakage-free setup)
# =====================================================================
# Stratifying by target 'y' preserves the ~22.75% return class ratio in both splits.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Identify numerical and categorical features
num_cols = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given"
]

cat_cols = ["product_category", "payment_method"]

# Define preprocessing pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine preprocessing using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ]
)

# Fit on training data ONLY, then transform both splits to prevent leakage
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

print("=== Preprocessing Dimensions ===")
print(f"Preprocessed X_train shape: {X_train_preprocessed.shape}")
print(f"Preprocessed X_test shape: {X_test_preprocessed.shape}\n")

# =====================================================================
# TASK 4: Baseline Dummy Classifier (most-frequent strategy)
# =====================================================================
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_clf.fit(X_train_preprocessed, y_train)

# Predict on test split
y_pred = dummy_clf.predict(X_test_preprocessed)

# Calculate and report metrics
accuracy = accuracy_score(y_test, y_pred)
f1_returned = f1_score(y_test, y_pred, pos_label=1)

print("=== Dummy Classifier Evaluation ===")
print(f"Accuracy: {accuracy:.6f}")
print(f"F1-score (returned=1): {f1_returned:.6f}")
print("\nClassification Report (Zero-Division Handled):")
print(classification_report(y_test, y_pred, zero_division=0))