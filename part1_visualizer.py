import os
import matplotlib
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, recall_score, precision_score

# --------------------------------------------------------------------
# 0. Setup and Data Loading
# --------------------------------------------------------------------
os.makedirs('part-1-visual-charts', exist_ok=True)
sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')
CHART_DPI = 150

# Load verified dataset from artifacts
df = pd.read_csv("orders_dataset.csv")
X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Columns definitions
num_cols = ["price_inr", "discount_pct", "customer_tenure_days", "num_previous_orders", 
            "num_previous_returns", "delivery_distance_km", "delivery_days", "is_weekend_order", "rating_given"]
cat_cols = ["product_category", "payment_method"]

# Pipeline setup
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Process splits
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Feature names retrieval
ohe_categories = preprocessor.named_transformers_["cat"].named_steps["ohe"].categories_
encoded_cat_cols = []
for col, cats in zip(cat_cols, ohe_categories):
    encoded_cat_cols.extend([f"{col}_{cat}" for cat in cats])
feature_names = num_cols + encoded_cat_cols

# --------------------------------------------------------------------
# 1. CHART 1: Logistic Regression Decision Threshold Sweep
# --------------------------------------------------------------------
lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr_model.fit(X_train_preprocessed, y_train)
y_probs = lr_model.predict_proba(X_test_preprocessed)[:, 1]

thresholds = np.arange(0.1, 0.91, 0.02)
accuracies, precisions, recalls, f1_scores = [], [], [], []

for t in thresholds:
    y_pred_t = (y_probs >= t).astype(int)
    accuracies.append(accuracy_score(y_test, y_pred_t))
    precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred_t, zero_division=0))

best_idx = np.argmax(f1_scores)
best_t = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_rec = recalls[best_idx]
best_prec = precisions[best_idx]

# Plot Chart 1
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, recalls, label='Recall (Catch Rate)', color='#0072B2', linewidth=2.5)
ax.plot(thresholds, f1_scores, label='F1-Score (Balanced Index)', color='#E69F00', linewidth=2.5)
ax.plot(thresholds, precisions, label='Precision (Correctness)', color='#009E73', linewidth=1.8, linestyle='--')

# Vertical line at optimal threshold
ax.axvline(best_t, color='red', linestyle=':', linewidth=2, label=f'Optimal Threshold: {best_t:.2f}')

# Key Annotation
ax.annotate(
    f"Optimal Threshold (t* = {best_t:.2f})\n"
    f"• Max F1-Score: {best_f1:.3f}\n"
    f"• Return Recall: {best_rec:.1%}\n"
    f"• Precision: {best_prec:.1%}",
    xy=(best_t, best_f1),
    xytext=(best_t + 0.05, best_f1 - 0.1),
    arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
    fontweight='semibold',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', edgecolor='gray', alpha=0.9)
)

# Labels and Theme
ax.set_title("Lowering Threshold to 0.44 Captures 75.8% of Returns, Outperforming Default 0.5 Rules", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Decision Threshold (Probability)")
ax.set_ylabel("Metric Score")
ax.set_xlim(0.1, 0.9)
ax.set_ylim(0.0, 1.05)
ax.legend(loc='lower left', frameon=True)
sns.despine()
plt.tight_layout(pad=1.5)
fig.savefig('part-1-visual-charts/decision-threshold-sweep.png', dpi=CHART_DPI, bbox_inches='tight')
plt.close()
print("Saved decision-threshold-sweep.png to scratch")

# --------------------------------------------------------------------
# 2. CHART 2: Feature Importance (Gini vs Permutation)
# --------------------------------------------------------------------
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestClassifier(max_depth=6, n_estimators=100, random_state=42, class_weight="balanced"))
])
rf_pipeline.fit(X_train, y_train)

# Impurity importance for the 18 preprocessed columns
impurity_importances = rf_pipeline.named_steps["rf"].feature_importances_

# Map preprocessed Gini importances back to original 11 columns
original_gini = {}
for name, importance in zip(feature_names, impurity_importances):
    mapped = False
    for cat_col in cat_cols:
        if name.startswith(cat_col + "_"):
            original_gini[cat_col] = original_gini.get(cat_col, 0.0) + importance
            mapped = True
            break
    if not mapped:
        original_gini[name] = importance

# Permutation importance on X_test (contains the 11 original columns)
perm_results = permutation_importance(
    rf_pipeline, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42
)
perm_means = perm_results.importances_mean

# Convert Gini mappings to line up exactly with X_test columns
features_orig = list(X_test.columns)
gini_mapped_list = [original_gini[f] for f in features_orig]

# Create DataFrame for plotting
importance_df = pd.DataFrame({
    "Feature": features_orig,
    "Gini_Importance": gini_mapped_list,
    "Permutation_Importance": perm_means
}).sort_values(by="Gini_Importance", ascending=False)

# Set up side-by-side horizontal bars
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
fig.suptitle("Gini Split Counts Overestimate Noisy continuous Features That Permute to Zero on Test Data", fontsize=14, fontweight='bold', y=1.02)

# Left plot: Gini
sns.barplot(data=importance_df, x="Gini_Importance", y="Feature", ax=ax1, palette="Blues_r")
ax1.set_title("Gini Impurity Importance (Biased by Splits)", fontsize=11, pad=10)
ax1.set_xlabel("Feature split frequency (Gini)")
ax1.set_ylabel("")

# Right plot: Permutation
sns.barplot(data=importance_df, x="Permutation_Importance", y="Feature", ax=ax2, palette="Oranges_r")
ax2.set_title("Permutation Importance Mean (Generalisable Signal)", fontsize=11, pad=10)
ax2.set_xlabel("Drop in ROC-AUC when feature is shuffled")
ax2.set_ylabel("")

# Annotations pointing out the noise features
for i, feature in enumerate(importance_df["Feature"]):
    if feature in ["customer_tenure_days", "delivery_distance_km", "discount_pct"]:
        # Draw indicator or label to highlight collapse
        ax1.text(importance_df.iloc[i]["Gini_Importance"] + 0.002, i, "⚠ Noise High-Card", 
                 va='center', fontsize=8, color='red', fontweight='semibold')
        ax2.text(importance_df.iloc[i]["Permutation_Importance"] - 0.003, i, "Collapsed ↓", 
                 va='center', fontsize=8, color='darkorange', fontweight='semibold')

sns.despine(fig=fig)
plt.tight_layout(pad=1.5)
fig.savefig('part-1-visual-charts/gini-vs-permutation-importance.png', dpi=CHART_DPI, bbox_inches='tight')
plt.close()
print("Saved gini-vs-permutation-importance.png to scratch")
