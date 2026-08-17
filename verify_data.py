import pandas as pd
import numpy as np

# Load the generated dataset
df = pd.read_csv("orders_dataset.csv")

# 1. Total shape verification
rows, cols = df.shape
print("=== Dataset Dimensions ===")
print(f"Total Rows: {rows}")
print(f"Total Columns: {cols}\n")

# 2. Overall metrics
overall_return_rate = df["returned"].mean()
missing_ratings_pct = df["rating_given"].isna().mean()
print("=== Key Baseline Metrics ===")
print(f"Overall Return Rate: {overall_return_rate:.4%}")
print(f"Missing 'rating_given' Percentage: {missing_ratings_pct:.4%}\n")

# 3. Return rate by product_category
print("=== Return Rate by Product Category ===")
cat_summary = df.groupby("product_category").agg(
    total_orders=("returned", "count"),
    returned_orders=("returned", "sum"),
    return_rate=("returned", "mean")
).reset_index()
print(cat_summary.to_string(index=False))
print()

# 4. Return rate by payment_method
print("=== Return Rate by Payment Method ===")
pay_summary = df.groupby("payment_method").agg(
    total_orders=("returned", "count"),
    returned_orders=("returned", "sum"),
    return_rate=("returned", "mean")
).reset_index()
print(pay_summary.to_string(index=False))
print()

# 5. Missingness of rating_given by Payment Method
print("=== Missingness of rating_given by Payment Method ===")
missing_by_payment = df.groupby("payment_method").agg(
    total_orders=("rating_given", "count"), # non-null count
    missing_count=("rating_given", lambda x: x.isna().sum()),
    missing_pct=("rating_given", lambda x: x.isna().mean())
).reset_index()
print(missing_by_payment.to_string(index=False))
print()

# 6. Missingness analysis (COD vs Non-COD)
cod_mask = df["payment_method"] == "COD"
cod_missing_pct = df[cod_mask]["rating_given"].isna().mean()
non_cod_missing_pct = df[~cod_mask]["rating_given"].isna().mean()
print("=== Missingness Pattern Analysis ===")
print(f"Missing rate for COD orders: {cod_missing_pct:.4%}")
print(f"Missing rate for Non-COD orders: {non_cod_missing_pct:.4%}")
print(f"Missingness gap: {cod_missing_pct - non_cod_missing_pct:.4%}")