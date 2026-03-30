#!/usr/bin/env python3
"""
COG functional enrichment analysis of cluster-shifted genes.

Genes are "shifters" if their cluster assignment differs between LB and M63.
Fisher's exact test per COG category, with BH FDR correction.

Run from repo root:
    /usr/bin/python3 analysis/enrichment.py
"""

import json
import os
import sys
import gzip
import io
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
    print("WARNING: matplotlib not available, figures will not be saved.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVES_JSON   = os.path.join(REPO_ROOT, "docs", "data", "curves_data.json")
CACHE_DIR     = os.path.join(REPO_ROOT, "results", "cache")
OUT_DIR       = os.path.join(REPO_ROOT, "results", "enrichment")

COG_CSV_URL   = "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/cog-20.cog.csv"
COG_DEF_URL   = "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/cog-20.def.tab"
FEATURE_TABLE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
    "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_feature_table.txt.gz"
)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Utility: download with cache
# ---------------------------------------------------------------------------
def download_cached(url, cache_name, timeout=60):
    """Download url and cache to CACHE_DIR/cache_name. Return local path or None."""
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
        print(f"  [saved]  {local} ({len(data)//1024} KB)")
        return local
    except Exception as exc:
        print(f"  [WARNING] Download failed for {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 1: Parse curves_data.json → per-gene cluster assignments
# ---------------------------------------------------------------------------
def load_gene_clusters():
    with open(CURVES_JSON) as fh:
        d = json.load(fh)

    genes_list = d["genes"]  # list of dicts with gene, jw_id, LB, M63

    records = []
    for g in genes_list:
        gene  = g["gene"]
        jw_id = g.get("jw_id", "")
        lb_info = g.get("LB")
        m63_info = g.get("M63")

        # Only include genes present in both media
        if lb_info is None or m63_info is None:
            continue

        cluster_LB  = lb_info["cluster"]
        cluster_M63 = m63_info["cluster"]
        is_shifter  = (cluster_LB != cluster_M63)
        records.append({
            "gene": gene,
            "jw_id": jw_id,
            "cluster_LB": cluster_LB,
            "cluster_M63": cluster_M63,
            "shifter": is_shifter,
        })

    df = pd.DataFrame(records)
    n_shifters = df["shifter"].sum()
    n_total    = len(df)
    print(f"Genes in both media: {n_total}  |  shifters: {n_shifters}  |  non-shifters: {n_total - n_shifters}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Download COG annotations
# ---------------------------------------------------------------------------
def load_cog_definitions():
    """
    Return dict: cog_id -> (func_letters, description)
    from cog-20.def.tab (format: cog_id TAB func_cat TAB description TAB ...)
    """
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
                func_cat = parts[1].strip()   # single or multi-letter e.g. "K" or "KJ"
                cog_name = parts[2].strip()
                cog_def[cog_id] = (func_cat, cog_name)
    print(f"  Loaded {len(cog_def)} COG definitions.")
    return cog_def


def load_cog_csv(cog_def):
    """
    Return DataFrame with columns: locus_tag, cog_id, func_category
    parsed from cog-20.cog.csv (E. coli K-12 MG1655, GCF_000005845.2).

    Actual COG 2020 CSV format (no header, comma-separated):
      col0: locus_tag  (e.g. "b0154")
      col1: assembly   (e.g. "GCF_000005845.2")
      col2: protein_id (e.g. "NP_414696.1")
      col3: prot_len
      col4: query_range
      col5: subject_len
      col6: cog_id     (e.g. "COG0001")
      col7: cog_id_dup
      col8: membership_class
      col9: score
      col10: evalue
      col11: cog_len
      col12: cog_range
    The func_category for each COG ID is obtained from cog_def.
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
        print(f"  WARNING: could not parse COG CSV: {exc}")
        return None

    print(f"  COG CSV total rows: {len(df_cog)}")

    # Filter to E. coli K-12 MG1655: assembly GCF_000005845.2
    # Also accept rows where locus_tag starts with "b" (Blattner numbers)
    mask = (
        df_cog["assembly"].astype(str).str.strip() == "GCF_000005845.2"
    ) | (
        df_cog["locus_tag"].astype(str).str.match(r'^b\d{4}$')
    )
    df_ecoli = df_cog[mask].copy()
    print(f"  E. coli K-12 COG rows: {len(df_ecoli)}")

    if len(df_ecoli) == 0:
        print("  WARNING: no E. coli rows found in COG CSV.")
        return None

    # Add func_category by joining with cog_def
    df_ecoli["func_category"] = df_ecoli["cog_id"].map(
        lambda cid: cog_def.get(str(cid), ("S", "Unknown"))[0]
    )
    df_ecoli["cog_name"] = df_ecoli["cog_id"].map(
        lambda cid: cog_def.get(str(cid), ("S", "Unknown"))[1]
    )

    df_ecoli = df_ecoli[["locus_tag", "cog_id", "func_category", "cog_name"]].drop_duplicates()
    print(f"  Unique E. coli locus tags with COG: {df_ecoli['locus_tag'].nunique()}")
    return df_ecoli


def load_feature_table():
    """
    Return DataFrame: gene_name -> locus_tag (b-number) mapping.
    Source: NCBI RefSeq feature table for E. coli K-12 MG1655.
    """
    local = download_cached(FEATURE_TABLE_URL, "GCF_000005845.2_feature_table.txt.gz")
    if local is None:
        return None

    try:
        with gzip.open(local, "rt", encoding="utf-8", errors="replace") as fh:
            df = pd.read_csv(fh, sep="\t", low_memory=False, on_bad_lines="skip")
    except Exception as exc:
        print(f"  WARNING: could not read feature table: {exc}")
        return None

    print(f"  Feature table columns: {list(df.columns[:10])}")
    print(f"  Feature table rows: {len(df)}")

    # Keep only genes (CDS rows with a gene symbol)
    # Typical columns: # feature, class, assembly, assembly_unit, seq_type,
    #   chromosome, genomic_accession, start, end, strand, product_accession,
    #   non-redundant_refseq, related_accession, name, symbol, GeneID,
    #   locus_tag, feature_interval_length, product_length, attributes
    if "symbol" in df.columns and "locus_tag" in df.columns:
        gene_col = "symbol"
    elif "name" in df.columns and "locus_tag" in df.columns:
        gene_col = "name"
    else:
        print("  WARNING: could not identify gene-name column in feature table.")
        return None

    df_genes = df[df[gene_col].notna() & df["locus_tag"].notna()][
        [gene_col, "locus_tag"]
    ].drop_duplicates()
    df_genes = df_genes.rename(columns={gene_col: "gene"})
    df_genes["gene"] = df_genes["gene"].str.strip().str.lower()
    df_genes["locus_tag"] = df_genes["locus_tag"].str.strip()
    print(f"  Gene→locus_tag mappings: {len(df_genes)}")
    return df_genes


# ---------------------------------------------------------------------------
# Step 3: Map gene names to COG categories
# ---------------------------------------------------------------------------
def build_gene_cog_map(gene_list):
    """
    Returns dict: gene_name (lower) -> list of single-letter COG category strings.
    Falls back to "S" (unknown) when no mapping is found.
    """
    print("\n--- Building gene → COG mapping ---")

    cog_def    = load_cog_definitions()
    feature_df = load_feature_table()
    cog_df     = load_cog_csv(cog_def)

    gene_cog = {}

    if feature_df is not None and cog_df is not None:
        # Join: gene -> locus_tag -> cog
        merged = feature_df.merge(cog_df, on="locus_tag", how="left")
        for _, row in merged.iterrows():
            g = str(row["gene"]).lower()
            fc = str(row.get("func_category", "S"))
            if pd.isna(row.get("func_category")):
                fc = "S"
            # func_category can be multi-letter like "KJ"
            cats = [c for c in fc if c.isalpha()]
            if not cats:
                cats = ["S"]
            if g not in gene_cog:
                gene_cog[g] = []
            gene_cog[g].extend(cats)

        # Deduplicate
        gene_cog = {g: list(set(cats)) for g, cats in gene_cog.items()}
        print(f"  Mapped {len(gene_cog)} genes via feature table + COG CSV.")
    else:
        print("  WARNING: download(s) failed. Mapping all genes to category 'S' (unknown).")

    # For any unmapped genes, assign "S"
    mapped = 0
    for gene in gene_list:
        gl = gene.lower()
        if gl in gene_cog and gene_cog[gl] != ["S"]:
            mapped += 1
        elif gl not in gene_cog:
            gene_cog[gl] = ["S"]

    total = len(gene_list)
    print(f"  Genes with known COG: {mapped}/{total}  ({100*mapped/max(total,1):.1f}%)")
    return gene_cog


# ---------------------------------------------------------------------------
# Step 4 & 5: Fisher's exact test + BH FDR
# ---------------------------------------------------------------------------
def bh_correction(pvalues):
    """Benjamini-Hochberg FDR correction. Returns q-values array."""
    n = len(pvalues)
    order   = np.argsort(pvalues)
    pvals_s = np.array(pvalues)[order]
    qvals   = np.zeros(n)
    cummin  = np.inf
    for i in range(n - 1, -1, -1):
        q = pvals_s[i] * n / (i + 1)
        cummin = min(cummin, q)
        qvals[order[i]] = cummin
    return np.minimum(qvals, 1.0)


def run_enrichment(df_genes, gene_cog):
    """
    df_genes: DataFrame with columns [gene, shifter]
    gene_cog: dict gene_lower -> list of COG category letters

    Returns: DataFrame with enrichment results per COG category.
    """
    # Add COG categories to gene table (primary = first category listed)
    df = df_genes.copy()
    df["gene_lower"] = df["gene"].str.lower()

    # Collect all unique categories across all genes
    all_cats = set()
    for g in df["gene_lower"]:
        for c in gene_cog.get(g, ["S"]):
            all_cats.add(c)

    results = []
    n_shifters     = df["shifter"].sum()
    n_nonshifters  = (~df["shifter"]).sum()

    for cat in sorted(all_cats):
        # For each gene, check if it has this category
        def has_cat(gene_lower):
            return cat in gene_cog.get(gene_lower, ["S"])

        df["has_cat"] = df["gene_lower"].apply(has_cat)

        a = int((df["has_cat"] & df["shifter"]).sum())       # shifter, in cat
        b = int((df["has_cat"] & ~df["shifter"]).sum())      # non-shifter, in cat
        c = int((~df["has_cat"] & df["shifter"]).sum())      # shifter, not in cat
        d = int((~df["has_cat"] & ~df["shifter"]).sum())     # non-shifter, not in cat

        # 2x2 contingency table:
        #            in_cat  not_in_cat
        # shifter      a        c
        # non-shifter  b        d
        contingency = [[a, c], [b, d]]
        odds_ratio, pvalue = stats.fisher_exact(contingency, alternative="two-sided")

        results.append({
            "cog_category": cat,
            "n_shifters_in_cat":    a,
            "n_nonshifters_in_cat": b,
            "n_shifters_total":     int(n_shifters),
            "n_nonshifters_total":  int(n_nonshifters),
            "odds_ratio":           odds_ratio,
            "pvalue":               pvalue,
        })

    results_df = pd.DataFrame(results).sort_values("pvalue").reset_index(drop=True)
    results_df["fdr_qvalue"] = bh_correction(results_df["pvalue"].tolist())
    return results_df


# ---------------------------------------------------------------------------
# COG category descriptions (fallback if def file unavailable)
# ---------------------------------------------------------------------------
COG_CATEGORY_DESCRIPTIONS = {
    "J": "Translation, ribosomal structure and biogenesis",
    "A": "RNA processing and modification",
    "K": "Transcription",
    "L": "Replication, recombination and repair",
    "B": "Chromatin structure and dynamics",
    "D": "Cell cycle control, cell division, chromosome partitioning",
    "Y": "Nuclear structure",
    "V": "Defense mechanisms",
    "T": "Signal transduction mechanisms",
    "M": "Cell wall/membrane/envelope biogenesis",
    "N": "Cell motility",
    "Z": "Cytoskeleton",
    "W": "Extracellular structures",
    "U": "Intracellular trafficking, secretion, and vesicular transport",
    "O": "Post-translational modification, protein turnover, chaperones",
    "X": "Mobilome: prophages, transposons",
    "C": "Energy production and conversion",
    "G": "Carbohydrate transport and metabolism",
    "E": "Amino acid transport and metabolism",
    "F": "Nucleotide transport and metabolism",
    "H": "Coenzyme transport and metabolism",
    "I": "Lipid transport and metabolism",
    "P": "Inorganic ion transport and metabolism",
    "Q": "Secondary metabolites biosynthesis, transport and catabolism",
    "R": "General function prediction only",
    "S": "Function unknown",
}


def add_descriptions(results_df, cog_def):
    """Add cog_description column from cog_def or fallback dict."""
    def get_desc(cat):
        # Try to find in cog_def (keyed by cog_id, not category letter)
        # Use our fallback dict for category-level descriptions
        return COG_CATEGORY_DESCRIPTIONS.get(cat, "Unknown")

    results_df["cog_description"] = results_df["cog_category"].apply(get_desc)
    return results_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_enrichment(results_df, out_prefix):
    if not matplotlib_import_ok:
        return

    sig = results_df[results_df["fdr_qvalue"] < 0.1].copy()
    plot_df = results_df.head(15).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#d62728" if q < 0.1 else "#1f77b4" for q in plot_df["fdr_qvalue"]]
    bars = ax.barh(
        [f'{r["cog_category"]}: {r["cog_description"][:40]}' for _, r in plot_df.iterrows()],
        -np.log10(plot_df["pvalue"].clip(lower=1e-300)),
        color=colors,
    )
    ax.set_xlabel("-log10(p-value)")
    ax.set_title("COG category enrichment in cluster-shifting genes\n(red = FDR < 0.1)")
    ax.invert_yaxis()
    plt.tight_layout()

    fig.savefig(out_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(out_prefix + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_prefix}.pdf / .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== COG Enrichment Analysis ===\n")

    # 1. Load cluster assignments
    print("--- Loading cluster assignments ---")
    df_genes = load_gene_clusters()

    # 2. Build gene → COG mapping
    gene_list = df_genes["gene"].tolist()
    gene_cog  = build_gene_cog_map(gene_list)

    # 3. Load COG definitions (for descriptions)
    cog_def = load_cog_definitions()

    # 4. Run enrichment
    print("\n--- Running Fisher's exact tests ---")
    results_df = run_enrichment(df_genes, gene_cog)
    results_df = add_descriptions(results_df, cog_def)

    # Reorder columns
    results_df = results_df[[
        "cog_category", "cog_description",
        "n_shifters_in_cat", "n_nonshifters_in_cat",
        "n_shifters_total", "n_nonshifters_total",
        "odds_ratio", "pvalue", "fdr_qvalue",
    ]]

    # 5. Save results
    out_csv = os.path.join(OUT_DIR, "cog_enrichment.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved enrichment results: {out_csv}")
    print(results_df.head(10).to_string(index=False))

    # 6. Plot
    out_fig = os.path.join(OUT_DIR, "fig_enrichment")
    plot_enrichment(results_df, out_fig)

    # 7. Save gene-level COG assignments
    df_genes["gene_lower"] = df_genes["gene"].str.lower()
    df_genes["cog_categories"] = df_genes["gene_lower"].apply(
        lambda g: ",".join(gene_cog.get(g, ["S"]))
    )
    df_genes["primary_cog"] = df_genes["cog_categories"].apply(
        lambda x: x.split(",")[0] if x else "S"
    )
    df_genes["cog_description"] = df_genes["primary_cog"].apply(
        lambda c: COG_CATEGORY_DESCRIPTIONS.get(c, "Unknown")
    )
    gene_out = df_genes.drop(columns=["gene_lower"]).copy()
    gene_out_path = os.path.join(OUT_DIR, "gene_cog_assignments.csv")
    gene_out.to_csv(gene_out_path, index=False)
    print(f"Saved gene COG assignments: {gene_out_path}")

    sig_count = (results_df["fdr_qvalue"] < 0.1).sum()
    print(f"\nSignificant categories (FDR < 0.1): {sig_count}")


if __name__ == "__main__":
    main()
