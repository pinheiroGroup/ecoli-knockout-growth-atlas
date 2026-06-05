#!/usr/bin/env python3
"""Per-COG ML-downstream on the Keio model-free atlas — for every target
in (μ_max, λ_loglin, K_emp) × every medium (LB, M63).

Parallel to cog_ml_per_medium.py, but parameterised on the target column so
that the same Spearman + RF semantics (matching GUIbiont-repo/src/ml_downstream.jl)
apply to all three kinetic targets the upgraded Kinbiont log-linear
estimator now exposes.

Inputs:
    results/keio_loglin_results.csv   (must contain gr_loglin, lag_loglin,
                                       N_max_emp per (gene, medium))
    results/ml/cog_one_hot_features.csv   (re-used; produced by
                                           cog_ml_per_medium.py)

Outputs (under results/ml_modelfree/):
    ml_cog_{target}_{LB,M63}.csv          per-target ranked results
    ml_cog_{target}_medium_delta.csv      LB-vs-M63 Δρ / Δimportance
    fig_ml_cog_{target}.{png,pdf}         3-panel comparison figure

Run:
    /usr/bin/python analysis/cog_ml_per_medium_modelfree.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrichment import (  # noqa: E402
    COG_CATEGORY_DESCRIPTIONS,
    build_gene_cog_map,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(REPO, "results")
OLD  = os.path.join(RES, "ml")                  # μ_max RF artifacts live here
OUT  = os.path.join(RES, "ml_modelfree")
os.makedirs(OUT, exist_ok=True)

# RF hyperparameters — identical to GUIbiont-repo/src/ml_downstream.jl
RF_KW = dict(
    n_estimators=100,
    max_depth=5,
    max_samples=0.7,
    random_state=42,
    n_jobs=-1,
)

EXCLUDE_COG = {"Y"}

# (target column in keio_loglin_results.csv, display label, units string,
#  short slug used in output filenames)
TARGETS = [
    ("gr_loglin",  r"$\mu_{\max}$",        "h$^{-1}$", "mu"),
    ("lag_loglin", r"$\lambda_\mathrm{loglin}$", "h",  "lag"),
    ("N_max_emp",  r"$K_\mathrm{emp}$",    "OD",       "Kemp"),
]


# ────────────────────────────────────────────────────────────────────────────
# Load per-gene target values
# ────────────────────────────────────────────────────────────────────────────
def load_target_per_medium(target_col: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RES, "keio_loglin_results.csv"))
    df = df[df["converged"] == True]  # noqa: E712
    out = df.pivot_table(index="gene", columns="medium",
                         values=target_col, aggfunc="first")
    out.columns = [f"y_{c}" for c in out.columns]
    return out.reset_index().dropna(subset=["y_LB", "y_M63"])


def build_cog_one_hot(genes: list[str]) -> pd.DataFrame:
    """Re-uses the COG one-hot already saved by cog_ml_per_medium.py if it
    exists; otherwise builds it from scratch."""
    cached = os.path.join(OLD, "cog_one_hot_features.csv")
    if os.path.exists(cached):
        df = pd.read_csv(cached)
        # Filter to the gene set we actually have
        df = df[df["gene"].isin(genes)].reset_index(drop=True)
        return df
    gene_cog = build_gene_cog_map(genes)
    all_cats = sorted({c for cats in gene_cog.values() for c in cats}
                      - EXCLUDE_COG)
    rows = []
    for g in genes:
        gl = g.lower()
        cats = set(gene_cog.get(gl, ["S"]))
        rows.append({c: int(c in cats) for c in all_cats})
    df = pd.DataFrame(rows)
    df.insert(0, "gene", genes)
    return df


# ────────────────────────────────────────────────────────────────────────────
# Spearman + RF — exact GUIbiont semantics
# ────────────────────────────────────────────────────────────────────────────
def spearman_per_feature(X: np.ndarray, y: np.ndarray,
                         feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(feature_names):
        xj = X[:, j]
        mask = np.isfinite(xj) & np.isfinite(y)
        if mask.sum() < 3:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(xj[mask], y[mask])
        rows.append({"feature": name, "rho": float(rho), "p": float(p),
                     "n_with": int(xj[mask].sum())})
    return pd.DataFrame(rows)


def rf_importance(X: np.ndarray, y: np.ndarray,
                  feature_names: list[str]) -> pd.DataFrame:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xm, ym = X[mask], y[mask]
    if Xm.shape[0] < 10:
        return pd.DataFrame(columns=["feature", "importance"])
    rf = RandomForestRegressor(**RF_KW).fit(Xm, ym)
    return (pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True))


def run_ml(medium: str, df_y: pd.DataFrame, df_cog: pd.DataFrame) -> pd.DataFrame:
    joined = df_y[["gene", f"y_{medium}"]].merge(df_cog, on="gene", how="inner")
    feat_cols = [c for c in joined.columns if c not in ("gene", f"y_{medium}")]
    X = joined[feat_cols].to_numpy(dtype=float)
    y = joined[f"y_{medium}"].to_numpy(dtype=float)
    spear = spearman_per_feature(X, y, feat_cols)
    imp   = rf_importance(X, y, feat_cols)
    out = spear.merge(imp, on="feature", how="outer")
    out["medium"] = medium
    out["description"] = out["feature"].map(
        lambda c: COG_CATEGORY_DESCRIPTIONS.get(c, "Unknown"))
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# Plot per target
# ────────────────────────────────────────────────────────────────────────────
def plot_compare(lb: pd.DataFrame, m63: pd.DataFrame,
                 target_label: str, out_prefix: str) -> None:
    m = lb.merge(m63, on="feature", suffixes=("_lb", "_m63"))
    m["description"] = m["description_lb"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.8),
                             gridspec_kw=dict(width_ratios=[1.4, 1.4, 1.2]))

    # (a) Spearman ρ paired
    ax = axes[0]
    m_a = m.sort_values("rho_m63").reset_index(drop=True)
    y = np.arange(len(m_a))
    bar_h = 0.4
    ax.barh(y - bar_h/2, m_a["rho_lb"], bar_h, color="#2E5C8A",
            edgecolor="black", linewidth=0.3, label="LB")
    ax.barh(y + bar_h/2, m_a["rho_m63"], bar_h, color="#B95C50",
            edgecolor="black", linewidth=0.3, label="M63")
    ax.axvline(0, color="k", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(m_a["feature"], fontsize=8, fontweight="bold")
    ax.set_xlabel(f"Spearman ρ (COG one-hot vs {target_label})")
    ax.set_title(f"(a) Per-COG correlation with {target_label}",
                 fontsize=10, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    # (b) RF importance paired
    ax = axes[1]
    m_b = m.sort_values("importance_m63", ascending=True).reset_index(drop=True)
    y = np.arange(len(m_b))
    ax.barh(y - bar_h/2, m_b["importance_lb"], bar_h, color="#2E5C8A",
            edgecolor="black", linewidth=0.3, label="LB")
    ax.barh(y + bar_h/2, m_b["importance_m63"], bar_h, color="#B95C50",
            edgecolor="black", linewidth=0.3, label="M63")
    ax.set_yticks(y)
    ax.set_yticklabels(m_b["feature"], fontsize=8, fontweight="bold")
    ax.set_xlabel("RF impurity importance")
    ax.set_title(f"(b) Random-forest importance (predicting {target_label})",
                 fontsize=10, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    # (c) Medium-specific Δρ
    ax = axes[2]
    m_c = m.copy()
    m_c["delta_rho"] = m_c["rho_m63"] - m_c["rho_lb"]
    m_c = m_c.sort_values("delta_rho").reset_index(drop=True)
    y = np.arange(len(m_c))
    colors = ["#B95C50" if d < 0 else "#2E5C8A" for d in m_c["delta_rho"]]
    ax.barh(y, m_c["delta_rho"], color=colors,
            edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="k", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(m_c["feature"], fontsize=8, fontweight="bold")
    ax.set_xlabel("Δρ  =  ρ(M63)  −  ρ(LB)")
    ax.set_title("(c) Medium-specific COG signature\n"
                 "negative = stronger effect in M63",
                 fontsize=10, loc="left")

    xlim = ax.get_xlim()
    xmax = max(abs(xlim[0]), abs(xlim[1]))
    if xmax == 0:
        xmax = 0.01
    ax.set_xlim(-xmax * 1.05, xmax * 1.05)
    label_pad = xmax * 0.03
    for i, row in m_c.iterrows():
        bar_to_left = row["delta_rho"] < 0
        ax.text(
            +label_pad if bar_to_left else -label_pad,
            i,
            str(row["description_lb"])[:32],
            fontsize=6.8, va="center",
            ha="left" if bar_to_left else "right",
            color="#444",
        )

    fig.suptitle(
        f"Keio ML-downstream — COG categories vs {target_label} "
        f"(model-free Kinbiont log-linear target)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_prefix + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(out_prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_prefix}.png / .pdf")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    print("=== Keio ML-downstream: COG categories → "
          "(μ_max, λ_loglin, K_emp) per medium ===\n")

    for target_col, target_label, _units, slug in TARGETS:
        print(f"\n========= Target: {target_label}  ({target_col})  =========")
        df_y = load_target_per_medium(target_col)
        if df_y.empty:
            print(f"  No data for {target_col} — column missing? Skipping.")
            continue
        print(f"  Genes with values in both media: {len(df_y)}")

        df_cog = build_cog_one_hot(df_y["gene"].tolist())
        cat_cols = [c for c in df_cog.columns if c != "gene"]
        coverage = (df_cog[cat_cols].sum(axis=1) > 0).mean()
        print(f"  COG matrix: {len(df_cog)} genes × {len(cat_cols)} cats "
              f"({100 * coverage:.1f}% coverage)")

        print(f"  Running ML for LB …")
        res_lb  = run_ml("LB",  df_y, df_cog)
        print(f"    LB top 5 by importance:")
        print(res_lb.head(5)[["feature", "description", "rho", "p", "importance"]]
              .to_string(index=False))

        print(f"  Running ML for M63 …")
        res_m63 = run_ml("M63", df_y, df_cog)
        print(f"    M63 top 5 by importance:")
        print(res_m63.head(5)[["feature", "description", "rho", "p", "importance"]]
              .to_string(index=False))

        res_lb.to_csv(os.path.join(OUT, f"ml_cog_{slug}_LB.csv"),  index=False)
        res_m63.to_csv(os.path.join(OUT, f"ml_cog_{slug}_M63.csv"), index=False)

        m = res_lb.merge(res_m63, on="feature", suffixes=("_lb", "_m63"))
        m["delta_rho"] = m["rho_m63"] - m["rho_lb"]
        m["delta_imp"] = m["importance_m63"] - m["importance_lb"]
        m["description"] = m["description_lb"]
        m_sorted = m.sort_values("delta_rho").reset_index(drop=True)
        m_sorted.to_csv(os.path.join(OUT, f"ml_cog_{slug}_medium_delta.csv"),
                        index=False)

        plot_compare(res_lb, res_m63, target_label,
                     os.path.join(OUT, f"fig_ml_cog_{slug}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
