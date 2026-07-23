#!/usr/bin/env python3
"""Build per-gene mean curves for LB and M63 from the Keio atlas raw data."""

import logging
import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
RESULTS_DIR = "results"

# OD readings at or below this value are treated as non-informative baseline /
# evaporated wells / blank Excel cells (which read as 0).
# Fallback to a permissive threshold for no-growth detection in clustering.
MIN_VALID_OD = -10.01 # skips min valid od

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

os.makedirs(RESULTS_DIR, exist_ok=True)


def normalize_curve_id(x):
    """Return curve id in the same format used by curve columns, e.g. Curve00001.

    Excel reads numeric ids as floats ("5.0"), so we coerce via int(float(...))
    when the value is numeric. Non-numeric ids are zero-padded as-is.
    """
    s = str(x).strip()

    if s.startswith("Curve"):
        return s

    try:
        n = int(float(s))
    except ValueError:
        return "Curve" + s.zfill(5)

    return f"Curve{n:05d}"


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

log.info("Example metadata curve_ids: %s", meta["curve_id"].head(10).tolist())
log.info("Media in metadata:\n%s", meta["medium"].value_counts())


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

    log.info("")
    log.info("=" * 80)
    log.info("Processing medium: %s", medium)
    log.info("=" * 80)
    log.info("%s curve columns: %s", medium, curves.columns[:10].tolist())

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
        valid_idx = np.where((~np.isnan(series)) & (series > MIN_VALID_OD))[0]

        if valid_idx.size == 0:
            # Completely invalid curve: skip it.
            invalid_curves += 1
            invalid_curve_records.append({
                "medium": medium,
                "curve_id": curve_id,
                "gene_name": gene,
                "reason": f"no_valid_od_above_{MIN_VALID_OD}",
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

    log.info("%s: metadata rows = %d", medium, len(subset))
    log.info("%s: valid curves used = %d", medium, valid_curves)
    log.info("%s: missing curves = %d", medium, missing_curves)
    log.info("%s: invalid curves skipped = %d", medium, invalid_curves)

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
    log.info("%s: total NaN values in output = %d", medium, nan_count)

    out_path = f"{RESULTS_DIR}/keio_{medium.lower()}_gene_means.csv"
    out.to_csv(out_path, index=False)

    log.info("%s: %d genes written", medium, len(result) - 1)
    log.info("%s: output written to %s", medium, out_path)

    # -------------------------------------------------------------------------
    # Save reports
    # -------------------------------------------------------------------------

    if invalid_curve_records:
        invalid_curves_path = f"{RESULTS_DIR}/keio_{medium.lower()}_invalid_curves.csv"
        pd.DataFrame(invalid_curve_records).to_csv(invalid_curves_path, index=False)
        log.info("%s: invalid curve report written to %s", medium, invalid_curves_path)

    if skipped_genes:
        skipped_genes_path = f"{RESULTS_DIR}/keio_{medium.lower()}_skipped_genes.csv"
        pd.DataFrame(skipped_genes).to_csv(skipped_genes_path, index=False)
        log.info("%s: skipped gene report written to %s", medium, skipped_genes_path)
