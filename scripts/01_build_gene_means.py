#!/usr/bin/env python3
"""Build per-gene mean curves for LB and M63 from the Keio atlas raw data."""

import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def normalize_curve_id(x):
    """Return curve id in the same format used by curve columns, e.g. Curve00001."""
    x = str(x).strip()

    if x.startswith("Curve"):
        return x

    x = x.replace(".0", "")
    return "Curve" + x.zfill(5)


# =============================================================================
# Load metadata
# =============================================================================

meta = pd.read_excel(f"{DATA_DIR}/Curves_knockouts_media.xlsx", header=0)
meta.columns = [c.strip() for c in meta.columns]

# Expected columns:
# curve_id, jw_id, gene_name, gene_category, medium
meta["curve_id"] = meta.iloc[:, 0].apply(normalize_curve_id)
meta["gene_name"] = meta.iloc[:, 2].astype(str).str.strip()
meta["medium"] = meta.iloc[:, 4].astype(str).str.strip()

print("Example metadata curve_ids:")
print(meta["curve_id"].head(10).tolist())

print()
print("Media in metadata:")
print(meta["medium"].value_counts())


# =============================================================================
# Input files
# =============================================================================

FILES = {
    "LB": f"{DATA_DIR}/Growth_curves_LB.xlsx",
    "M63": f"{DATA_DIR}/Growth_curves_M63.xlsx",
}

# Load LB once, because M63 may have NaN times.
lb_curves_for_time = pd.read_excel(FILES["LB"])
lb_time_col = lb_curves_for_time.columns[0]
lb_times = lb_curves_for_time[lb_time_col].values.astype(float)


# =============================================================================
# Process each medium independently
# =============================================================================

for medium, path in FILES.items():
    curves = pd.read_excel(path)

    print()
    print("=" * 80)
    print(f"Processing medium: {medium}")
    print("=" * 80)

    print(f"{medium} curve columns:")
    print(curves.columns[:10].tolist())

    time_col = curves.columns[0]
    times = curves[time_col].values.astype(float)

    # If the medium file has NaN times, reuse the LB time axis.
    if np.isnan(times).all():
        times = lb_times.copy()

    subset = meta[meta["medium"] == medium]

    groups = {}

    missing_curves = 0
    valid_curves = 0
    invalid_curves = 0

    invalid_curve_records = []

    for _, row in subset.iterrows():
        curve_id = row["curve_id"]
        gene = row["gene_name"]

        if curve_id not in curves.columns:
            missing_curves += 1
            continue

        series = curves[curve_id].values.astype(float).copy()

        # Identify real observed OD values.
        # Values <= 0.01 are treated as non-informative baseline/artefact.
        valid_idx = np.where((~np.isnan(series)) & (series > 0.01))[0]

        if valid_idx.size == 0:
            # Completely invalid curve: skip it.
            invalid_curves += 1
            invalid_curve_records.append({
                "medium": medium,
                "curve_id": curve_id,
                "gene_name": gene,
                "reason": "no_valid_od_above_0.01",
            })
            continue

        first_valid = valid_idx[0]
        last_valid = valid_idx[-1]

        # Fill leading NaNs, if any, with the first valid value.
        series[:first_valid] = series[first_valid]

        # Forward-fill internal NaNs.
        s = pd.Series(series)
        s = s.ffill()
        series = s.values.astype(float)

        # Extend the curve after the last valid biological OD value.
        # This handles both trailing zeros and trailing NaNs.
        last_value = series[last_valid]
        series[last_valid + 1:] = last_value

        groups.setdefault(gene, []).append(series)
        valid_curves += 1

    print(f"{medium}: metadata rows = {len(subset)}")
    print(f"{medium}: valid curves used = {valid_curves}")
    print(f"{medium}: missing curves = {missing_curves}")
    print(f"{medium}: invalid curves skipped = {invalid_curves}")

    # -------------------------------------------------------------------------
    # Build per-gene mean curves
    # -------------------------------------------------------------------------

    result = {"Time": times}

    skipped_genes = []

    # Genes present in metadata for this medium
    expected_genes = sorted(subset["gene_name"].dropna().unique())

    for gene in expected_genes:
        reps = groups.get(gene, [])

        if len(reps) == 0:
            skipped_genes.append({
                "medium": medium,
                "gene_name": gene,
                "reason": "no_valid_replicates",
            })
            continue

        mat = np.vstack(reps)
        result[gene] = np.nanmean(mat, axis=0)

    out = pd.DataFrame(result)

    nan_count = out.isna().sum().sum()
    print(f"{medium}: total NaN values in output = {nan_count}")

    out_path = f"{RESULTS_DIR}/keio_{medium.lower()}_gene_means.csv"
    out.to_csv(out_path, index=False)

    print(f"{medium}: {len(result) - 1} genes written")
    print(f"{medium}: output written to {out_path}")

    # -------------------------------------------------------------------------
    # Save reports
    # -------------------------------------------------------------------------

    if invalid_curve_records:
        invalid_curves_path = f"{RESULTS_DIR}/keio_{medium.lower()}_invalid_curves.csv"
        pd.DataFrame(invalid_curve_records).to_csv(invalid_curves_path, index=False)
        print(f"{medium}: invalid curve report written to {invalid_curves_path}")

    if skipped_genes:
        skipped_genes_path = f"{RESULTS_DIR}/keio_{medium.lower()}_skipped_genes.csv"
        pd.DataFrame(skipped_genes).to_csv(skipped_genes_path, index=False)
        print(f"{medium}: skipped gene report written to {skipped_genes_path}")