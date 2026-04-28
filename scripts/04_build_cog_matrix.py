#!/usr/bin/env python3
"""Build a gene × COG-category multi-hot feature matrix."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from ml_keio import build_gene_cog_map
import pandas as pd

df = pd.read_csv("results/keio_lb_gene_means.csv")
if "gene" in df.columns:
    genes = df["gene"].dropna().astype(str).unique().tolist()
else:
    genes = [col for col in df.columns if col != "Time"]

gene_cog = build_gene_cog_map(genes)
all_cats = sorted({c for cats in gene_cog.values() for c in cats})

rows = [
    {"gene": g, **{f"COG_{c}": int(c in gene_cog.get(g.lower(), [])) for c in all_cats}}
    for g in genes
]
pd.DataFrame(rows).to_csv("results/cog_feature_matrix.csv", index=False)
print(f"Saved: {len(rows)} genes × {len(all_cats)} COG categories")
