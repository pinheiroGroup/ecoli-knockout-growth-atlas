# Reproducing the Keio knockout analysis with GUIbiont

This guide walks through replicating the clustering, batch fitting, and ML
analysis from the paper using the GUIbiont web interface.  
Two short Python scripts handle data preparation; everything else runs in the browser.

**Assumed paths** (adjust if your setup differs):

| Path | Contents |
|---|---|
| `data/` | Raw Excel files and `all_curves.csv` |
| `results/` | Output directory |
| `$GUIBIONT` | Root of the GUIbiont server directory |

---

## Prerequisites

- GUIbiont server running (`julia --project=. --threads=auto web_server.jl` from the GUIbiont repo)
- Python ≥ 3.9 with `pandas`, `numpy`, `openpyxl`
- Install dependencies once: `pip install pandas numpy openpyxl`
- Julia ≥ 1.12, with the analysis environment instantiated once:

  ```bash
  cd analysis
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```

  This installs the exact versions recorded in `analysis/Manifest.toml`
  (Kinbiont v1.5.1, from the General registry). Skipping it makes Step 3 fail
  at precompile with `Package Parsers ... is required but does not seem to be
  installed`. The first run pulls a large dependency tree and takes a while.

---

## Step 1 — Build per-gene mean curves

The raw data contains individual replicate curves keyed by curve ID.
This script reads the metadata mapping, aggregates replicates by gene,
and outputs one CSV per medium ready to load into GUIbiont.

Run from the repo root:

```bash
python scripts/01_build_gene_means.py
```

Output: `results/keio_lb_gene_means.csv`, `results/keio_m63_gene_means.csv`  
Format: `Time, geneA, geneB, ...` — one column per gene, rows = time points.

---

## Step 2 — Register as GUIbiont experiments

GUIbiont's Batch Fit tab reads from `$GUIBIONT/Clean_data/<experiment>/`.
This script creates the two experiment directories with the correct file names.

Run from the repo root:

```bash
python scripts/02_register_experiments.py
```

---

## Step 3 — Clustering

The manuscript's clustering (Guibiont.tex/SM.tex Methods, "Trajectory
preparation for clustering") runs on the **complete 0.25–50 h gene-mean
trajectories from Step 1 directly — no interpolation, no truncated grid**.
`analysis/analyse.jl` reproduces this natively against `Kinbiont.jl` (the same
`preprocess`/`FitOptions` call GUIbiont's `/api/cluster` route makes), so it
never drifts from what the GUIbiont interface itself would compute:

```bash
cd analysis
julia --project=. analyse.jl
```

This loads the raw workbooks, builds the full-length gene means, runs the
non-growing pre-screen (constant-curve criterion, τ=0.5, q=0.05/0.95, **no**
trend test — `Kinbiont`'s `cluster_trend_test` defaults to `true` and must be
passed as `false` explicitly, or it reserves extra sentinel slots beyond the
pre-screen), and clusters with **k=2 for LB, k=3 for M63** (WCSS elbow
baseline at k=1 with the pre-screen disabled, applied for k≥2 — Guibiont.tex
"Elbow support for choosing k"). Output: `docs/data/curves_data.json`, plus
the centroid tables for Figure 2c if `GUIBIONT_PAPER_SCRIPTS_DIR` points at
the paper repo's `scripts/` directory.

To confirm the run reproduced the manuscript, check `docs/data/curves_data.json`:
`optimal_k` should be `{"LB": 2, "M63": 3}` and `nongrowing_genes` should hold
0 genes for LB and 97 for M63. Clustering is seeded (`kmeans_seed = 42`), so a
correct run reproduces the committed `curves_data.json` and the two centroid
tables byte for byte. If your elbow or centroids move between runs, you are on
a checkout that predates commit `fe169cf` — pull and re-instantiate.

If you'd rather drive this from the GUIbiont interface directly instead
(equivalent, just manual): **Clustering** tab → **File** mode → upload
`results/keio_lb_gene_means.csv` → leave **Interpolate to common grid**
unchecked → pre-screen on, τ=0.5 → cluster method `kmeans`, k=2 (k=3 for
M63). Do **not** enable the trend test.

---

## Step 4 — Batch fitting

The manuscript uses the **log-linear sliding-window estimator only** — not
the four-model AICc parametric fit. `analysis/run_keio_loglin_via_guibiont.py`
drives this through GUIbiont's actual `/api/batch-fit-loglin` endpoint (the
same route the Batch Fit tab's "Log-linear only" option calls), with the
manuscript's parameters baked in (`LOGLIN_PARAMS` in the script):

```bash
python analysis/run_keio_loglin_via_guibiont.py
```

Output: `results/keio_loglin_results.csv`, 7,770 gene×medium rows. Expect
7,767 converged (`leuA`, `lysA` and `nuoB` in M63 return no finite positive
$\mu_{\max}$ — SM.tex, "Additional log-linear results").

Equivalently from the interface: **Batch Fit** tab → experiment `keio_lb` →
**Log-linear only** → Run → Download CSV. Repeat for `keio_m63`.

The four-model parametric fit (`analysis/batch_fit.jl`,
`scripts/05_clean_fit_results.py`) is **not** part of the published Keio
analysis; it predates the log-linear methodology and is kept only for the
COG-based exploratory analysis below, which is likewise not in the paper.

---

## Step 5 — Identify cluster shifters (external script)

Genes whose cluster assignment differs between LB and M63, read from the
clustering output of Step 3:

```bash
python scripts/03_find_shifters.py
```

---

## Step 6 — COG functional enrichment (external script)

**Steps 6–8 are supplementary exploratory analysis, not part of the
published manuscript** (which reports KEGG pathway enrichment among the
non-growing strains, via `analysis/kegg_enrichment_s2.py`, not COG
categories — see Supplementary Data S2). They still depend on the four-model
parametric fit from Step 4's `batch_fit.jl`, which the paper no longer uses.

Uses `enrichment.py` from the atlas repo unchanged.  
It reads `results/cluster_shifters.csv` and downloads COG annotations from NCBI.

```bash
python analysis/enrichment.py
```

Output: `results/enrichment/` — CSVs with odds ratio, p-value, FDR per COG category.

---

## Step 7 — Build COG feature matrix

One-time script to build the multi-hot COG matrix needed by the ML Analysis tab.

Run `scripts/04_build_cog_matrix.py`:

```bash
python scripts/04_build_cog_matrix.py
```

Also prepare cleaned per-medium fit CSVs (filter converged, cap lag):

```bash
python scripts/05_clean_fit_results.py
```

---

## Step 8 — ML Analysis (GUIbiont interface)

Do this **twice** — once for LB, once for M63.

1. Open the **ML Analysis** tab
2. **Fit results CSV**: upload `results/keio_lb_batch_fit_clean.csv`
3. **Label column**: `gene`
4. **Feature matrix CSV**: upload `results/lb_cog_feature_matrix.csv` (use `m63_cog_feature_matrix.csv` for the M63 run)
5. **Parameters to analyse**: select `gr`, `N_max`, `lag`
6. Click **Run**

The interface shows:
- **Spearman correlations** — ranked bar chart per parameter
- **Random forest feature importance** — top COG categories per parameter
- **Partial dependence plots** — marginal effect of top 5 COG categories

Repeat with `keio_m63_batch_fit_clean.csv` for M63.

---

## Summary of files produced

| File | How |
|---|---|
| `results/keio_lb_gene_means.csv` | Script 01 |
| `results/keio_m63_gene_means.csv` | Script 01 |
| `results/clusters_lb.csv` | GUIbiont Clustering tab |
| `results/clusters_m63.csv` | GUIbiont Clustering tab |
| `results/keio_lb_batch_fit.csv` | GUIbiont Batch Fit tab |
| `results/keio_m63_batch_fit.csv` | GUIbiont Batch Fit tab |
| `results/cluster_assignments_both.csv` | Script 03 |
| `results/cluster_shifters.csv` | Script 03 |
| `results/enrichment/` | `enrichment.py` |
| `results/{lb,m63}_cog_feature_matrix.csv` | Script 04 |
| `results/keio_{lb,m63}_batch_fit_clean.csv` | Script 05 |
