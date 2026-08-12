#!/usr/bin/env python3
"""Clean GUIbiont batch fit exports for ML Analysis upload.

NOT used to reproduce the manuscript's results: cleans the four-model AICc
batch fit (batch_fit.jl), which the paper does not use (it fits gr/N_max
with the log-linear estimator only). Feeds the COG-based exploratory
analysis only.
"""

import numpy as np
import pandas as pd


RENAME_COLUMNS = {
    "well": "gene",
    "model": "best_model",
    "aic": "aicc",
    "exit_lag_rate": "lag",
}


def clean_fit_results(path):
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in RENAME_COLUMNS.items() if k in df.columns})

    missing = {"gene", "gr", "N_max", "lag"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(sorted(missing))}")

    if "converged" not in df.columns:
        df["converged"] = True

    df = df[df["converged"].astype(str).str.lower() == "true"].copy()
    df.loc[df["lag"] >= 49.9, "lag"] = np.nan
    return df


for medium in ["lb", "m63"]:
    src = f"results/keio_{medium}_batch_fit.csv"
    dst = f"results/keio_{medium}_batch_fit_clean.csv"
    cleaned = clean_fit_results(src)
    cleaned.to_csv(dst, index=False)
    print(f"{medium.upper()}: {len(cleaned)} converged rows -> {dst}")
