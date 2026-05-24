"""
Population Stability Index (PSI) drift monitor.
Detects when the score distribution has shifted enough to warrant retraining.
"""
import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute PSI between a reference score distribution and a new one.
    PSI < 0.1  → stable
    PSI 0.1–0.2 → minor shift, monitor
    PSI > 0.2  → significant shift, retrain
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def feature_psi_report(train_df: pd.DataFrame, prod_df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """PSI for each numeric feature — flags which features are drifting."""
    records = []
    for col in numeric_cols:
        if col in train_df.columns and col in prod_df.columns:
            p = psi(train_df[col].dropna().values, prod_df[col].dropna().values)
            records.append({"feature": col, "psi": round(p, 4),
                            "status": "stable" if p < 0.1 else "monitor" if p < 0.2 else "DRIFT"})
    return pd.DataFrame(records).sort_values("psi", ascending=False)