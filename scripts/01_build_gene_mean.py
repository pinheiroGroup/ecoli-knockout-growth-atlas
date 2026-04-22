#!/usr/bin/env python3
"""Build per-gene mean curves for LB and M63 from the Keio atlas raw data."""
import numpy as np
import pandas as pd

DATA_DIR    = "data"
RESULTS_DIR = "results"

# --- Load metadata: curve_id → (gene_name, medium) ---
meta = pd.read_excel(f"{DATA_DIR}/Curves_knockouts_media.xlsx", header=0)
# Expected columns: curve_id, jw_id, gene_name, gene_category, medium
meta.columns = [c.strip() for c in meta.columns]
meta["curve_id"]   = meta.iloc[:, 0].astype(str).str.strip()
meta["gene_name"]  = meta.iloc[:, 2].astype(str).str.strip()
meta["medium"]     = meta.iloc[:, 4].astype(str).str.strip()

# --- Load all curves (row index + Time column + Curve##### columns) ---
curves_m63 = pd.read_excel(f"{DATA_DIR}/Growth_curves_M63.xlsx")
curves_lb = pd.read_excel(f"{DATA_DIR}/Growth_curves_LB.xlsx")
curves = pd.concat([curves_lb, curves_m63.iloc[:,1:]])
curves.to_csv(f"{DATA_DIR}/all_curves.csv")

time_col = curves.columns[0]
times    = curves[time_col].values.astype(float)

for medium in ["LB", "M63"]:
    subset   = meta[meta["medium"] == medium]
    gene_map = dict(zip(subset["curve_id"], subset["gene_name"]))

    groups: dict[str, list] = {}
    for curve_id, gene in gene_map.items():
        col = f"Curve{curve_id}"
        if col not in curves.columns:
            col = curve_id
        if col not in curves.columns:
            continue
        groups.setdefault(gene, []).append(
            curves[col].values.astype(float)
        )

    result = {"Time": times}
    for gene, reps in groups.items():
        mat = np.vstack(reps)           # shape: n_reps × n_tp
        result[gene] = np.nanmean(mat, axis=0)

    out = pd.DataFrame(result)
    out.to_csv(f"{RESULTS_DIR}/keio_{medium.lower()}_gene_means.csv", index=False)
    print(f"{medium}: {len(result) - 1} genes written")

