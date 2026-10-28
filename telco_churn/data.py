"""Deterministic validation before any fitted transformation."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
               "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
               "Contract", "PaperlessBilling", "PaymentMethod"]
COLUMNS = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
           "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
           "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"]
SOURCE_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def distribution(y):
    return {"negative": int((y == 0).sum()), "positive": int((y == 1).sum()),
            "positive_fraction": float((y == 1).mean())}

def clean_frame(raw):
    if raw.columns.duplicated().any() or set(raw.columns) != set(COLUMNS):
        raise ValueError("Expected exactly the 21 Telco columns")
    df = raw.copy()
    missing = {c: int((df[c].isna() | df[c].astype(str).str.strip().eq("")).sum()) for c in COLUMNS}
    duplicates = int(df.customerID.duplicated().sum())
    if duplicates or missing["customerID"]:
        raise ValueError("customerID must be present and unique")
    if set(df.Churn.unique()) != {"No", "Yes"}:
        raise ValueError("Churn must contain No and Yes only")
    if not df.SeniorCitizen.isin([0, 1]).all():
        raise ValueError("SeniorCitizen must be binary 0/1")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    invalid = df.TotalCharges.isna()
    # No blanket dropna: other bad columns fail visibly, including in removed rows.
    for c in COLUMNS:
        if c != "TotalCharges" and (missing[c] or df[c].isna().any()):
            raise ValueError(f"Unexpected missing/invalid values in {c}")
    if not np.isfinite(df.loc[~invalid, NUMERIC].to_numpy()).all():
        raise ValueError("Numeric inputs must be finite")
    cleaned = df.loc[~invalid].copy()
    y = cleaned.Churn.map({"No": 0, "Yes": 1}).astype(int)
    metadata = {"raw_rows": len(df), "column_count": len(df.columns), "columns": list(raw.columns),
                "removed_rows": int(invalid.sum()), "retained_rows": len(cleaned),
                "target_classes": ["No", "Yes"], "duplicate_customerID_count": duplicates,
                "missing_or_blank_before_conversion": missing,
                "TotalCharges_invalid_after_conversion": int(invalid.sum()),
                "raw_distribution": distribution(df.Churn.map({"No": 0, "Yes": 1})),
                "cleaned_distribution": distribution(y)}
    return cleaned.drop(columns=["customerID", "Churn"]), y, metadata

def load_data(path):
    raw = pd.read_csv(path)
    X, y, meta = clean_frame(raw)
    meta.update({"sha256": digest(path), "file_size_bytes": Path(path).stat().st_size})
    return X, y, meta

def split_data(X, y, seed=42):
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
