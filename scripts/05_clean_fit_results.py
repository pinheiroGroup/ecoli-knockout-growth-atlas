#!/usr/bin/env python3
"""Clean GUIbiont batch fit exports for ML Analysis upload."""

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
    src = f"results/keio_batch_fit_{medium}.csv"
    dst = f"results/keio_batch_fit_{medium}_clean.csv"
    cleaned = clean_fit_results(src)
    cleaned.to_csv(dst, index=False)
    print(f"{medium.upper()}: {len(cleaned)} converged rows -> {dst}")
