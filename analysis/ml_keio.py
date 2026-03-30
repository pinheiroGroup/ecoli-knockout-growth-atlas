#!/usr/bin/env python3
"""
Random forest (or Spearman correlation fallback) to predict growth parameters
from COG functional category.

Run from repo root:
    /usr/bin/python3 analysis/ml_keio.py
"""

import gzip
import os
import sys
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
from scipy import stats

matplotlib_import_ok = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_import_ok = False
    print("WARNING: matplotlib not available; figures will not be saved.")

sklearn_ok = True
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.inspection import permutation_importance
except ImportError:
    sklearn_ok = False
    print("WARNING: sklearn not available; falling back to Spearman correlations.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIT_CSV     = os.path.join(REPO_ROOT, "results", "keio_batch_fit_results.csv")
CACHE_DIR   = os.path.join(REPO_ROOT, "results", "cache")
OUT_DIR     = os.path.join(REPO_ROOT, "results", "ml_keio")

FEATURE_TABLE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
    "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_feature_table.txt.gz"
)
COG_CSV_URL = "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/cog-20.cog.csv"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# COG category descriptions
# ---------------------------------------------------------------------------
COG_CATEGORY_DESCRIPTIONS = {
    "J": "Translation",
    "A": "RNA processing",
    "K": "Transcription",
    "L": "DNA replication/repair",
    "B": "Chromatin structure",
    "D": "Cell cycle/division",
    "V": "Defense mechanisms",
    "T": "Signal transduction",
    "M": "Cell wall/membrane",
    "N": "Cell motility",
    "U": "Intracellular trafficking",
    "O": "Post-transl. modification",
    "X": "Mobilome",
    "C": "Energy production",
    "G": "Carbohydrate metabolism",
    "E": "Amino acid metabolism",
    "F": "Nucleotide metabolism",
    "H": "Coenzyme metabolism",
    "I": "Lipid metabolism",
    "P": "Inorganic ion transport",
    "Q": "Secondary metabolites",
    "R": "General function",
    "S": "Unknown function",
}

# ---------------------------------------------------------------------------
# Utility: download with cache
# ---------------------------------------------------------------------------
def download_cached(url, cache_name, timeout=60):
    local = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(local):
        print(f"  [cache] {cache_name}")
        return local
    print(f"  [download] {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        with open(local, "wb") as fh:
            fh.write(data)
        print(f"  [saved] {local} ({len(data)//1024} KB)")
        return local
    except Exception as exc:
        print(f"  [WARNING] Download failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Gene → COG mapping (same logic as enrichment.py, self-contained)
# ---------------------------------------------------------------------------
def load_feature_table():
    local = download_cached(FEATURE_TABLE_URL, "GCF_000005845.2_feature_table.txt.gz")
    if local is None:
        return None
    try:
        with gzip.open(local, "rt", encoding="utf-8", errors="replace") as fh:
            df = pd.read_csv(fh, sep="\t", low_memory=False, on_bad_lines="skip")
    except Exception as exc:
        print(f"  WARNING: feature table read failed: {exc}")
        return None

    if "symbol" in df.columns and "locus_tag" in df.columns:
        gene_col = "symbol"
    elif "name" in df.columns and "locus_tag" in df.columns:
        gene_col = "name"
    else:
        print("  WARNING: no gene-name column found in feature table.")
        return None

    df_g = df[df[gene_col].notna() & df["locus_tag"].notna()][[gene_col, "locus_tag"]].drop_duplicates()
    df_g = df_g.rename(columns={gene_col: "gene"})
    df_g["gene"]      = df_g["gene"].str.strip().str.lower()
    df_g["locus_tag"] = df_g["locus_tag"].str.strip()
    print(f"  Feature table: {len(df_g)} gene→locus_tag mappings.")
    return df_g


def load_cog_definitions():
    """Return dict: cog_id -> (func_letters, description)"""
    COG_DEF_URL = "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/cog-20.def.tab"
    local = download_cached(COG_DEF_URL, "cog-20.def.tab")
    if local is None:
        return {}
    cog_def = {}
    with open(local, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                cog_id   = parts[0].strip()
                func_cat = parts[1].strip()
                cog_name = parts[2].strip()
                cog_def[cog_id] = (func_cat, cog_name)
    print(f"  Loaded {len(cog_def)} COG definitions.")
    return cog_def


def load_cog_csv(cog_def):
    """
    Return DataFrame: locus_tag, func_category for E. coli K-12.
    Actual COG 2020 CSV format (no header, comma-separated):
      col0: locus_tag, col1: assembly, col2: protein_id, col3: prot_len,
      col4: query_range, col5: subject_len, col6: cog_id, col7: cog_id_dup,
      col8: membership, col9: score, col10: evalue, col11: cog_len, col12: cog_range
    """
    local = download_cached(COG_CSV_URL, "cog-20.cog.csv")
    if local is None:
        return None

    col_names = [
        "locus_tag", "assembly", "protein_id", "prot_len",
        "query_range", "subject_len", "cog_id", "cog_id_dup",
        "membership_class", "score", "evalue", "cog_len", "cog_range"
    ]
    try:
        df_cog = pd.read_csv(local, header=None, names=col_names,
                             low_memory=False, on_bad_lines="skip")
    except Exception as exc:
        print(f"  WARNING: COG CSV parse failed: {exc}")
        return None

    mask = (
        df_cog["assembly"].astype(str).str.strip() == "GCF_000005845.2"
    ) | (
        df_cog["locus_tag"].astype(str).str.match(r'^b\d{4}$')
    )
    df_ecoli = df_cog[mask].copy()

    if len(df_ecoli) == 0:
        print("  WARNING: no E. coli rows found in COG CSV.")
        return None

    # Add func_category from cog_def
    df_ecoli["func_category"] = df_ecoli["cog_id"].map(
        lambda cid: cog_def.get(str(cid), ("S", "Unknown"))[0]
    )
    df_ecoli = df_ecoli[["locus_tag", "func_category"]].drop_duplicates()
    print(f"  COG CSV: {len(df_ecoli)} E. coli K-12 rows.")
    return df_ecoli


def build_gene_cog_map(gene_list):
    """Return dict: gene_lower -> list of COG category letters."""
    print("\n--- Building gene → COG mapping ---")
    cog_def    = load_cog_definitions()
    feature_df = load_feature_table()
    cog_df     = load_cog_csv(cog_def)

    gene_cog = {}
    if feature_df is not None and cog_df is not None:
        merged = feature_df.merge(cog_df, on="locus_tag", how="left")
        for _, row in merged.iterrows():
            g  = str(row["gene"]).lower()
            fc = str(row.get("func_category", "S"))
            if pd.isna(row.get("func_category")):
                fc = "S"
            cats = [c for c in fc if c.isalpha()] or ["S"]
            gene_cog.setdefault(g, set()).update(cats)
        gene_cog = {g: list(cats) for g, cats in gene_cog.items()}
        mapped = sum(1 for g in gene_list if g.lower() in gene_cog and gene_cog[g.lower()] != ["S"])
        print(f"  Mapped {mapped}/{len(gene_list)} genes with known COG.")
    else:
        print("  WARNING: falling back to 'S' for all genes.")

    for gene in gene_list:
        gl = gene.lower()
        if gl not in gene_cog:
            gene_cog[gl] = ["S"]

    return gene_cog


# ---------------------------------------------------------------------------
# Build feature matrix: multi-hot COG categories
# ---------------------------------------------------------------------------
def build_feature_matrix(df, gene_cog, all_cats):
    """
    For each row (gene), create a multi-hot vector over all_cats.
    Returns np.ndarray of shape (n_rows, n_cats).
    """
    cat_index = {c: i for i, c in enumerate(all_cats)}
    n = len(df)
    X = np.zeros((n, len(all_cats)), dtype=np.float32)
    for i, gene in enumerate(df["gene"].str.lower()):
        for c in gene_cog.get(gene, ["S"]):
            if c in cat_index:
                X[i, cat_index[c]] = 1.0
    return X


# ---------------------------------------------------------------------------
# Analysis: random forest or Spearman fallback
# ---------------------------------------------------------------------------
def analyze_parameter(df_medium, gene_cog, all_cats, param, medium):
    """
    Run RF or Spearman on a single (param, medium) combination.
    Returns DataFrame of feature importances sorted descending.
    """
    # Filter: converged == True, valid param
    df_sub = df_medium.copy()
    df_sub = df_sub[df_sub["converged"].astype(str).str.lower() == "true"]
    df_sub = df_sub[df_sub[param].notna()]

    # Exclude lag at upper bound (non-convergence indicator)
    if param == "lag":
        df_sub = df_sub[df_sub[param] < 49.9]

    if len(df_sub) < 10:
        print(f"  Skipping {param}/{medium}: only {len(df_sub)} valid rows.")
        return None

    X = build_feature_matrix(df_sub, gene_cog, all_cats)
    y = df_sub[param].values.astype(np.float64)

    # Drop feature columns with zero variance
    variances   = X.var(axis=0)
    valid_cols  = variances > 0
    X_filtered  = X[:, valid_cols]
    cats_filtered = [c for c, v in zip(all_cats, valid_cols) if v]

    if X_filtered.shape[1] == 0:
        print(f"  Skipping {param}/{medium}: no variable features.")
        return None

    importances = np.zeros(len(cats_filtered))

    if sklearn_ok:
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_filtered, y)
        importances = rf.feature_importances_
        try:
            scores = cross_val_score(rf, X_filtered, y, cv=5, scoring="r2")
            r2_cv = scores.mean()
        except Exception:
            r2_cv = np.nan
        print(f"  RF {param}/{medium}: n={len(df_sub)}, CV R²={r2_cv:.3f}")
        method = "RF_importance"
    else:
        # Spearman correlation per COG column
        for j in range(X_filtered.shape[1]):
            col = X_filtered[:, j]
            if col.std() == 0:
                continue
            r, _ = stats.spearmanr(col, y)
            importances[j] = abs(r) if not np.isnan(r) else 0.0
        print(f"  Spearman {param}/{medium}: n={len(df_sub)}")
        method = "Spearman_abs_rho"

    imp_df = pd.DataFrame({
        "cog_category": cats_filtered,
        "importance":   importances,
        "method":       method,
        "parameter":    param,
        "medium":       medium,
        "n_genes":      len(df_sub),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df["cog_description"] = imp_df["cog_category"].map(
        lambda c: COG_CATEGORY_DESCRIPTIONS.get(c, "Unknown")
    )
    return imp_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_importance(imp_lb, imp_m63, param, out_prefix):
    if not matplotlib_import_ok:
        return
    if imp_lb is None and imp_m63 is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    top_n = 15

    for ax, imp_df, medium in zip(axes, [imp_lb, imp_m63], ["LB", "M63"]):
        if imp_df is None:
            ax.set_visible(False)
            continue
        top = imp_df.head(top_n)
        labels = [f'{r["cog_category"]}: {r["cog_description"][:30]}' for _, r in top.iterrows()]
        ax.barh(labels, top["importance"], color="#2196F3")
        ax.invert_yaxis()
        method = top["method"].iloc[0] if len(top) > 0 else ""
        ax.set_xlabel(method)
        ax.set_title(f"{param} — {medium}  (n={top['n_genes'].iloc[0] if len(top)>0 else 0})")

    fig.suptitle(f"COG category importance for {param}")
    plt.tight_layout()
    fig.savefig(out_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(out_prefix + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_prefix}.pdf / .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== ML / COG feature importance analysis ===\n")

    # Load fit results
    df_fit = pd.read_csv(FIT_CSV)
    print(f"Loaded fit results: {df_fit.shape[0]} rows")
    print(f"Columns: {list(df_fit.columns)}")

    # Build gene → COG map
    genes_all = df_fit["gene"].dropna().unique().tolist()
    gene_cog  = build_gene_cog_map(genes_all)

    # Collect all COG categories present
    all_cats = sorted(set(
        c for cats in gene_cog.values() for c in cats
    ))
    print(f"\nCOG categories present: {all_cats}")

    parameters = ["gr", "N_max", "lag"]
    media      = ["LB", "M63"]

    all_importances = {}

    for param in parameters:
        if param not in df_fit.columns:
            print(f"  WARNING: column '{param}' not found, skipping.")
            continue
        imp_lb  = None
        imp_m63 = None
        for medium in media:
            df_med = df_fit[df_fit["medium"] == medium].copy()
            imp = analyze_parameter(df_med, gene_cog, all_cats, param, medium)
            if imp is not None:
                out_csv = os.path.join(OUT_DIR, f"feature_importance_{param}_{medium}.csv")
                imp.to_csv(out_csv, index=False)
                print(f"  Saved: {out_csv}")
                all_importances[(param, medium)] = imp
            if medium == "LB":
                imp_lb = imp
            else:
                imp_m63 = imp

        # Save importance figure for each parameter
        out_fig = os.path.join(OUT_DIR, f"fig_importance_{param}")
        plot_importance(imp_lb, imp_m63, param, out_fig)

    # Summary figure: gr LB vs M63 side by side (main figure requested)
    imp_gr_lb  = all_importances.get(("gr", "LB"))
    imp_gr_m63 = all_importances.get(("gr", "M63"))
    out_main_fig = os.path.join(OUT_DIR, "fig_importance")
    plot_importance(imp_gr_lb, imp_gr_m63, "gr", out_main_fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
