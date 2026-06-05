#!/usr/bin/env python3
"""Run GUIbiont's ML-downstream (impurity + permutation + CV R² + Spearman)
on the Keio COG one-hot screen through /api/ml-downstream and persist the
canonical CSVs, per medium.

API-driven replacement for cog_ml_per_medium_modelfree.py (which used
sklearn locally). Every number now comes from the same DecisionTree.jl
forest the GUIbiont UI exposes — sqrt-p subsampling, 0.7 partial sample,
100 trees, max depth 5.

Inputs:
    results/keio_loglin_results.csv     (model-free targets per gene)
    results/ml/cog_one_hot_features.csv (24-COG one-hot per gene)

Outputs (results/ml_modelfree/):
    ml_cog_{mu,lag,Kemp}_{LB,M63}.csv         # impurity ranking
    perm_cog_{mu,lag,Kemp}_{LB,M63}.csv       # permutation ranking
    correlations_cog_{LB,M63}.csv             # Spearman ρ per COG × target
    cv_r2_summary.csv                          # CV R² over all six runs

Run:
    /usr/bin/python analysis/run_cog_ml_via_guibiont.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE  = Path(__file__).resolve().parent
REPO  = HERE.parent
RES   = REPO / "results"
OUT   = RES / "ml_modelfree"
OUT.mkdir(exist_ok=True)

LOGLIN_CSV = RES / "keio_loglin_results.csv"
COG_CSV    = RES / "ml" / "cog_one_hot_features.csv"

API = os.environ.get("GUIBIONT_API", "http://localhost:9090")
# (target column on the loglin CSV → short slug used in output filenames)
TARGETS = [("gr",         "mu"),
           ("lag_loglin", "lag"),
           ("N_max_emp",  "Kemp")]
MEDIA = ("LB", "M63")


def main() -> None:
    df  = pd.read_csv(LOGLIN_CSV)
    df  = df[df["converged"] == True].copy()
    cog = pd.read_csv(COG_CSV)

    cv_rows = []
    for medium in MEDIA:
        sub = df[df["medium"] == medium][
            ["gene", "gr_loglin", "lag_loglin", "N_max_emp"]
        ].rename(columns={"gr_loglin": "gr"})

        payload = dict(
            fit_csv        = sub.to_csv(index=False),
            label_col      = "gene",
            feature_matrix = cog.to_csv(index=False),
            params         = [t for t, _ in TARGETS],
        )
        req = urllib.request.Request(
            f"{API}/api/ml-downstream", method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode())
        with urllib.request.urlopen(req, timeout=900) as r:
            body = json.loads(r.read())

        print(f"[{medium}] n_wells = {body['n_wells']}")

        # Spearman ρ — per medium, one row per COG, columns per target.
        corr_rows = []
        for row in body["correlations"]:
            out = {"COG": row["feature"]}
            for t, _ in TARGETS:
                out[t] = row.get(t)
            corr_rows.append(out)
        pd.DataFrame(corr_rows).to_csv(
            OUT / f"correlations_cog_{medium}.csv", index=False)

        for target_col, slug in TARGETS:
            imp = pd.DataFrame(body["importance"][target_col])
            imp = imp.rename(columns={"feature": "COG"})
            imp.to_csv(OUT / f"ml_cog_{slug}_{medium}.csv", index=False)

            perm = pd.DataFrame(body["permutation_importance"][target_col])
            perm = perm.rename(columns={"feature": "COG"})
            perm.to_csv(OUT / f"perm_cog_{slug}_{medium}.csv", index=False)

            cv = body["cv_r2"][target_col]
            cv_rows.append(dict(
                dataset="Keio", medium=medium, target=slug,
                cv_r2_mean=cv["mean"], cv_r2_std=cv["std"], n=cv["n"],
            ))

            print(f"  {slug:>5}: impurity {imp.iloc[0]['COG']} "
                  f"({imp.iloc[0]['importance']:.3f}) | "
                  f"perm {perm.iloc[0]['COG']} "
                  f"({perm.iloc[0]['permutation_importance']:.3f}) | "
                  f"CV R² = {cv['mean']:+.3f}")

    pd.DataFrame(cv_rows).to_csv(OUT / "cv_r2_summary.csv", index=False)
    print(f"\nwrote {OUT / 'cv_r2_summary.csv'}")


if __name__ == "__main__":
    main()
