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
# curves has two halves: rows 0–199 (LB, valid time) and rows 200–399
# (M63, NaN time). Both media share the same 200-point plate reader schedule
# (0.25–50 h in 0.25 h steps), so we reuse the LB time axis for M63.
curves_m63 = pd.read_excel(f"{DATA_DIR}/Growth_curves_M63.xlsx")
curves_lb = pd.read_excel(f"{DATA_DIR}/Growth_curves_LB.xlsx")
curves = pd.concat([curves_lb, curves_m63.iloc[:,1:]])
curves.to_csv(f"{DATA_DIR}/all_curves.csv")


time_col = curves.columns[0]
all_times = curves[time_col].values.astype(float)

shared_times = all_times[~np.isnan(all_times)]   # 200 time points, 0.25–50 h

# Row ranges for each medium in the combined CSV
MEDIUM_ROWS = {"LB": slice(0, 200), "M63": slice(200, 400)}

for medium in ["LB", "M63"]:
    subset   = meta[meta["medium"] == medium]
    gene_map = dict(zip(subset["curve_id"], subset["gene_name"]))
    row_slice = MEDIUM_ROWS[medium]

    groups: dict[str, list] = {}
    for curve_id, gene in gene_map.items():
        col = f"Curve{curve_id}"
        if col not in curves.columns:
            col = curve_id
        if col not in curves.columns:
            continue
        series = curves[col].values.astype(float)[row_slice]
        # Mask trailing zeros: zeros after the last OD > 0.01 are artefacts
        # (blank Excel cells read as 0, or evaporated/failed wells)
        valid_idx = np.where(series > 0.01)[0]
        if valid_idx.size > 0:
            series[valid_idx[-1] + 1:] = np.nan
        groups.setdefault(gene, []).append(series)

    result = {"Time": shared_times}
    for gene, reps in groups.items():
        mat = np.vstack(reps)           # shape: n_reps × 200
        result[gene] = np.nanmean(mat, axis=0)

    out = pd.DataFrame(result)
    out.to_csv(f"{RESULTS_DIR}/keio_{medium.lower()}_gene_means.csv", index=False)
    print(f"{medium}: {len(result) - 1} genes written")
