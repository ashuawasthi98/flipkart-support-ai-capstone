import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, classification_report
)
from sklearn.inspection import permutation_importance

def run_part1():
    print("="*60)
    print("STARTING PART 1: RETURN-RISK SCORING PIPELINE")
    print("="*60)
    
    # -------------------------------------------------------------
    # TASK 1: Generate the exact seeded dataset
    # -------------------------------------------------------------
    print("\n--- Task 1: Generating Seeded Dataset ---")
    rng = np.random.default_rng(42)
    N = 6000

    categories = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
    cat_probs = [0.32, 0.22, 0.18, 0.18, 0.10]
    payment_methods = ["COD", "Prepaid_Card", "Prepaid_UPI", "Wallet"]
    pay_probs = [0.42, 0.24, 0.24, 0.10]

    product_category = rng.choice(categories, size=N, p=cat_probs)
    payment_method = rng.choice(payment_methods, size=N, p=pay_probs)

    base_price = {
        "Apparel": (400, 2200),
        "Electronics": (1200, 45000),
        "Home": (300, 8000),
        "Footwear": (500, 4500),
        "Beauty": (150, 2500),
    }
    price_inr = np.round(np.array([rng.uniform(*base_price[c]) for c in product_category]), 0)

    discount_pct = np.clip(rng.normal(22, 15, N), 0, 75)
    customer_tenure_days = np.clip(rng.exponential(380, N), 1, 2500).round(0)
    num_previous_orders = np.clip((customer_tenure_days / 45) + rng.normal(0, 2, N), 0, None).round(0)
    base_return_rate = np.clip(rng.beta(1.5, 9, N), 0, 1)
    num_previous_returns = np.round(base_return_rate * num_previous_orders).clip(0, num_previous_orders)

    delivery_distance_km = np.clip(rng.gamma(3, 90, N), 2, 2200).round(1)
    delivery_days = np.clip(rng.normal(4.5, 2.2, N), 1, 21).round(0)
    is_weekend_order = rng.integers(0, 2, N)

    rating_given = rng.integers(1, 6, N).astype(float)
    missing_mask = rng.random(N) < np.where(payment_method == "COD", 0.22, 0.06)
    rating_given[missing_mask] = np.nan

    fit_risk_cat = np.isin(product_category, ["Apparel", "Footwear"]).astype(float)
    prev_return_ratio = np.where(num_previous_orders > 0, num_previous_returns / np.maximum(num_previous_orders, 1), 0)

    z = (-2.2 
         + 1.9 * prev_return_ratio 
         + 0.55 * fit_risk_cat 
         + 0.014 * (discount_pct - 20) / 10 
         + 0.9 * (payment_method == "COD").astype(float) 
         + 0.10 * (delivery_days - 4.5) / 2 
         + 0.30 * (price_inr / base_price["Electronics"][1]) 
         + 0.05 * is_weekend_order 
         - 0.15 * np.tanh(customer_tenure_days / 500))

    prob_return = 1 / (1 + np.exp(-z))
    returned = (rng.random(N) < prob_return).astype(int)

    df = pd.DataFrame({
        "order_id": np.arange(1, N + 1),
        "product_category": product_category,
        "price_inr": price_inr,
        "discount_pct": np.round(discount_pct, 1),
        "payment_method": payment_method,
        "customer_tenure_days": customer_tenure_days.astype(int),
        "num_previous_orders": num_previous_orders.astype(int),
        "num_previous_returns": num_previous_returns.astype(int),
        "delivery_distance_km": delivery_distance_km,
        "delivery_days": delivery_days.astype(int),
        "is_weekend_order": is_weekend_order,
        "rating_given": rating_given,
        "returned": returned,
    })

    df.to_csv("orders_dataset.csv", index=False)
    print(f"Generated orders_dataset.csv | Shape: {df.shape}")
    print(f"Overall Return Rate: {df['returned'].mean():.4%}")
    print(f"Missing rating_given: {df['rating_given'].isna().mean():.4%}")

    # -------------------------------------------------------------
    # TASK 2: Verify the generated data
    # -------------------------------------------------------------
    print("\n--- Task 2: Data Verification Tables ---")
    print("\n1. Return Rate by Category:")
    cat_summary = df.groupby("product_category").agg(
        total_orders=("returned", "count"),
        returned_orders=("returned", "sum"),
        return_rate=("returned", "mean")
    ).reset_index()
    print(cat_summary.to_string(index=False))

    print("\n2. Return Rate by Payment Method:")
    pay_summary = df.groupby("payment_method").agg(
        total_orders=("returned", "count"),
        returned_orders=("returned", "sum"),
        return_rate=("returned", "mean")
    ).reset_index()
    print(pay_summary.to_string(index=False))

    print("\n3. Missingness by Payment Method:")
    missing_by_pay = df.groupby("payment_method").agg(
        total_orders=("rating_given", "count"),
        missing_count=("rating_given", lambda x: x.isna().sum()),
        missing_pct=("rating_given", lambda x: x.isna().mean())
    ).reset_index()
    print(missing_by_pay.to_string(index=False))

    # -------------------------------------------------------------
    # TASK 3: Preprocess without leakage
    # -------------------------------------------------------------
    print("\n--- Task 3: Building leakage-free Preprocessing Pipeline ---")
    X = df.drop(columns=["order_id", "returned"])
    y = df["returned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    num_cols = [
        "price_inr", "discount_pct", "customer_tenure_days", 
        "num_previous_orders", "num_previous_returns", 
        "delivery_distance_km", "delivery_days", "is_weekend_order", "rating_given"
    ]
    cat_cols = ["product_category", "payment_method"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )

    # -------------------------------------------------------------
    # TASK 4: Build a baseline
    # -------------------------------------------------------------
    print("\n--- Task 4: Dummy Baseline Model ---")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train_preprocessed, y_train)
    y_pred_dummy = dummy.predict(X_test_preprocessed)
    
    print(f"Dummy Test Accuracy: {accuracy_score(y_test, y_pred_dummy):.6f}")
    print(f"Dummy Test F1 (class 1): {f1_score(y_test, y_pred_dummy, pos_label=1, zero_division=0):.6f}")

    # -------------------------------------------------------------
    # TASK 5: Train and tune a Logistic Regression model
    # -------------------------------------------------------------
    print("\n--- Task 5: Logistic Regression and Threshold Sweep ---")
    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X_train_preprocessed, y_train)

    y_prob_lr_test = lr.predict_proba(X_test_preprocessed)[:, 1]

    # Default 0.5 Threshold
    y_pred_lr_05 = (y_prob_lr_test >= 0.5).astype(int)
    print("\nLogistic Regression at Default 0.5 Threshold:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_lr_05):.6f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_lr_05):.6f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_lr_05):.6f}")
    print(f"  Precision: {precision_score(y_test, y_pred_lr_05):.6f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob_lr_test):.6f}")

    # Sweep
    thresholds = np.arange(0.1, 0.901, 0.02)
    lr_sweep = []
    for t in thresholds:
        y_pred_t = (y_prob_lr_test >= t).astype(int)
        lr_sweep.append({
            "Threshold": round(t, 2),
            "F1": f1_score(y_test, y_pred_t, zero_division=0),
            "Recall": recall_score(y_test, y_pred_t, zero_division=0),
            "Precision": precision_score(y_test, y_pred_t, zero_division=0),
            "Accuracy": accuracy_score(y_test, y_pred_t)
        })
    df_lr_sweep = pd.DataFrame(lr_sweep)
    idx_best_lr = df_lr_sweep["F1"].idxmax()
    best_lr = df_lr_sweep.loc[idx_best_lr]

    print(f"\nOptimal Logistic Regression Threshold: {best_lr['Threshold']:.2f}")
    print(f"  F1-Score:  {best_lr['F1']:.6f}")
    print(f"  Recall:    {best_lr['Recall']:.6f}")
    print(f"  Precision: {best_lr['Precision']:.6f}")
    print(f"  Accuracy:  {best_lr['Accuracy']:.6f}")
    print(f"  Recall Increase vs Default: {best_lr['Recall'] - recall_score(y_test, y_pred_lr_05):+.6%}")
    print(f"  Precision Drop vs Default:  {best_lr['Precision'] - precision_score(y_test, y_pred_lr_05):+.6%}")

    # -------------------------------------------------------------
    # TASK 6: Train and tune a Random Forest model
    # -------------------------------------------------------------
    print("\n--- Task 6: GridSearchCV Random Forest ---")
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("rf", RandomForestClassifier(class_weight="balanced", random_state=42))
    ])

    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [6, 10, None]
    }

    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_rf_pipeline = grid_search.best_estimator_
    best_rf_params = grid_search.best_params_
    best_rf_cv_auc = grid_search.best_score_

    y_prob_rf_test = best_rf_pipeline.predict_proba(X_test)[:, 1]
    rf_test_auc = roc_auc_score(y_test, y_prob_rf_test)

    print(f"Best Parameters: {best_rf_params}")
    print(f"Best CV ROC-AUC: {best_rf_cv_auc:.6f}")
    print(f"Test Set ROC-AUC: {rf_test_auc:.6f}")
    print(f"ROC-AUC Gap (CV - Test): {best_rf_cv_auc - rf_test_auc:.6f}")

    # -------------------------------------------------------------
    # TASK 7: Explain the model
    # -------------------------------------------------------------
    print("\n--- Task 7: Feature Importance vs Permutation Importance ---")
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    all_features = num_cols + cat_names

    rf_model = best_rf_pipeline.named_steps["rf"]
    impurity_importances = rf_model.feature_importances_

    perm_result = permutation_importance(
        rf_model, X_test_preprocessed, y_test, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1
    )

    df_explain = pd.DataFrame({
        "Feature": all_features,
        "Impurity_Importance": impurity_importances,
        "Permutation_Mean": perm_result.importances_mean,
    })
    df_explain["Impurity_Rank"] = df_explain["Impurity_Importance"].rank(ascending=False).astype(int)
    df_explain["Permutation_Rank"] = df_explain["Permutation_Mean"].rank(ascending=False).astype(int)

    print("\nComparison Table (Sorted by Impurity-Based Importance):")
    print(df_explain.sort_values(by="Impurity_Rank").head(10).to_string(index=False))

    # -------------------------------------------------------------
    # TASK 8: Subgroup / root-cause analysis
    # -------------------------------------------------------------
    print("\n--- Task 8: Subgroup Analysis on Random Forest ---")
    # Finding F1-maximizing threshold on RF on test split (Task 9 requirement done early)
    rf_sweep = []
    for t in thresholds:
        y_pred_t = (y_prob_rf_test >= t).astype(int)
        rf_sweep.append({
            "Threshold": round(t, 2),
            "F1": f1_score(y_test, y_pred_t, zero_division=0),
            "Recall": recall_score(y_test, y_pred_t, zero_division=0),
            "Precision": precision_score(y_test, y_pred_t, zero_division=0),
            "Accuracy": accuracy_score(y_test, y_pred_t)
        })
    df_rf_sweep = pd.DataFrame(rf_sweep)
    idx_best_rf = df_rf_sweep["F1"].idxmax()
    best_rf = df_rf_sweep.loc[idx_best_rf]
    t_star_rf = best_rf["Threshold"]

    print(f"\nRandom Forest F1-maximizing Threshold (t*_rf): {t_star_rf:.2f}")
    print(f"  F1-Score:  {best_rf['F1']:.6f}")
    print(f"  Recall:    {best_rf['Recall']:.6f}")
    print(f"  Precision: {best_rf['Precision']:.6f}")
    print(f"  Accuracy:  {best_rf['Accuracy']:.6f}")

    y_pred_rf = (y_prob_rf_test >= t_star_rf).astype(int)

    eval_df = X_test.copy()
    eval_df["returned_true"] = y_test
    eval_df["returned_pred"] = y_pred_rf

    print("\nSubgroups by Product Category:")
    cat_sub = []
    for cat in eval_df["product_category"].unique():
        sub = eval_df[eval_df["product_category"] == cat]
        cat_sub.append({
            "Product Category": cat,
            "Total": len(sub),
            "Returns": sub["returned_true"].sum(),
            "Recall": recall_score(sub["returned_true"], sub["returned_pred"], zero_division=0),
            "Precision": precision_score(sub["returned_true"], sub["returned_pred"], zero_division=0)
        })
    print(pd.DataFrame(cat_sub).to_string(index=False))

    print("\nSubgroups by Payment Method:")
    pay_sub = []
    for pay in eval_df["payment_method"].unique():
        sub = eval_df[eval_df["payment_method"] == pay]
        pay_sub.append({
            "Payment Method": pay,
            "Total": len(sub),
            "Returns": sub["returned_true"].sum(),
            "Recall": recall_score(sub["returned_true"], sub["returned_pred"], zero_division=0),
            "Precision": precision_score(sub["returned_true"], sub["returned_pred"], zero_division=0)
        })
    print(pd.DataFrame(pay_sub).to_string(index=False))

    # -------------------------------------------------------------
    # TASK 9: Save the final pipeline
    # -------------------------------------------------------------
    print("\n--- Task 9: Saving Winning Pipeline ---")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_rf_pipeline, "models/return_risk_model.pkl")
    print("Successfully saved fitted Pipeline to models/return_risk_model.pkl")
    
    # Save a copy in out/ directly for delivery
    joblib.dump(best_rf_pipeline, "return_risk_model.pkl")
    df.to_csv("orders_dataset.csv", index=False)
    
    print("\n" + "="*60)
    print("PART 1 PIPELINE EXECUTED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    run_part1()
