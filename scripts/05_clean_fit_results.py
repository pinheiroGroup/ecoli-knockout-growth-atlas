#!/usr/bin/env python3
"""Clean batch fit results for ML Analysis upload."""
import numpy as np, pandas as pd

for medium in ["lb", "m63"]:
    df = pd.read_csv(f"results/keio_batch_fit_{medium}.csv")
    df = df[df["converged"].astype(str).str.lower() == "true"].copy()
    df.loc[df["lag"] >= 49.9, "lag"] = np.nan
    df.to_csv(f"results/keio_batch_fit_{medium}_clean.csv", index=False)
    print(f"{medium.upper()}: {len(df)} converged rows")
