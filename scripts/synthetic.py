"""Synthetic fixtures: no external dataset/network dependency."""
import numpy as np
import pandas as pd
from telco_churn.data import CATEGORICAL, COLUMNS

def make_frame(n=160, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(["A", "B", "C"], n) for c in CATEGORICAL})
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["customerID"] = [f"synthetic-{i}" for i in range(n)]
    df["tenure"] = rng.integers(1, 73, n)
    df["MonthlyCharges"] = rng.uniform(20, 120, n)
    df["TotalCharges"] = (df.tenure * df.MonthlyCharges).astype(str)
    df["Churn"] = np.where((df.tenure < 22) ^ (rng.random(n) < .12), "Yes", "No")
    return df[COLUMNS]
