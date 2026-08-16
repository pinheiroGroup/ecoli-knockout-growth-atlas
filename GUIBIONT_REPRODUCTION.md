# Reproducing the Keio knockout analysis with GUIbiont

This guide walks through the complete command-line reproduction of the
clustering, batch fitting and KEGG enrichment reported in the paper, with the
equivalent GUIbiont browser steps noted where applicable.

**Assumed paths** (adjust if your setup differs):

| Path | Contents |
|---|---|
| `data/` | Raw Excel workbooks from the published dataset: `Growth_curves_LB.xlsx`, `Growth_curves_M63.xlsx`, `Curves_knockouts_media.xlsx`, `Growth_parameters.xlsx`, `Bacterial stocks.xlsx` |
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

Run from the repo root, passing the GUIbiont repo path as the first argument:

```bash
python scripts/02_register_experiments.py /path/to/GUIbiont
```

The path can also be supplied through the `GUIBIONT_DIR` environment variable:

```bash
GUIBIONT_DIR=/path/to/GUIbiont python scripts/02_register_experiments.py
```

Use one of these two forms on a headless machine. With neither the argument nor
the environment variable set, the script falls back to a tkinter folder picker
and then to an `input()` prompt, which may block or fail without a display or
an interactive terminal.

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

From the repository root, refresh the source-code tab shown on GitHub Pages:

```bash
python scripts/sync_docs_analysis_source.py
```

This loads the raw workbooks, builds the full-length gene means, runs the
non-growing pre-screen (constant-curve criterion, τ=0.5, q=0.05/0.95, **no**
trend test — `Kinbiont`'s `cluster_trend_test` defaults to `true` and must be
passed as `false` explicitly, or it reserves extra sentinel slots beyond the
pre-screen), and clusters with **k=2 for LB, k=3 for M63** (WCSS elbow
baseline at k=1 with the pre-screen disabled, applied for k≥2 — Guibiont.tex
"Elbow support for choosing k"). Output: `docs/data/curves_data.json`, plus
the centroid tables for Figure 2c if `GUIBIONT_PAPER_SCRIPTS_DIR` points at
the paper repo's `scripts/` directory. It also writes
`results/keio_s1_metadata.csv`, containing the `jw_id` and replicate count for
each gene–medium mean used by Supplementary Data S1, and
`results/keio_m63_nongrowing_genes.json`, a deterministic `{"genes": [...]}`
adapter containing the M63 pre-screen class used by the KEGG analysis.

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

### Step 3b — KEGG enrichment of the non-growing class

After Step 3, run from the repository root:

```bash
python analysis/kegg_enrichment_s2.py
```

By default the script reads the generated
`results/keio_m63_nongrowing_genes.json`; no manual extraction from
`docs/data/curves_data.json` is required. A different compatible JSON can
still be supplied with `--non-growers PATH`. The command writes
`results/keio_kegg_pathway_memberships.csv` and
`results/keio_nongrowing_enrichment.csv`, the two inputs used for
Supplementary Data S2.

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

**On `analysis/batch_fit.jl` (the four-model parametric fit).** The
*parametric fit itself* — selecting among logistic / Gompertz / Baranyi /
aHPM by AICc — is **not** what the manuscript reports. It predates the
log-linear methodology, and none of its fitted parameters are published: Step 4
above (log-linear) is what produces the paper's actual growth-rate results.

The script is retained because it supports the optional COG/ML exploration and
is a useful four-model fitting example. It is **not required to reproduce the
manuscript or Supplementary Data S1**. Step 3 writes the two aggregation fields
needed by S1 (`jw_id` and `n_replicates`) directly to
`results/keio_s1_metadata.csv`, before any parametric model is fitted. Run the
following only if you want the exploratory parametric output:

```bash
julia --threads auto --project=analysis analysis/batch_fit.jl
```

`scripts/05_clean_fit_results.py`, which post-processes that CSV for the COG
exploratory analysis, is likewise not part of the manuscript (see Steps 6–8).

---

## Step 5 — Identify cluster shifters (external script)

**Feeds only into the exploratory COG analysis below (Steps 6–8); not part of
the published manuscript.**

Genes whose cluster assignment differs between LB and M63.

This script reads `results/clusters_lb.csv` and `results/clusters_m63.csv`,
which are the per-medium **Download CSV** exports from the GUIbiont Clustering
tab — not the JSON that Step 3 writes. Produce them from the interface first
(Clustering tab → run LB, download; repeat for M63), otherwise the script exits
with `FileNotFoundError: results/clusters_lb.csv`.

```bash
python scripts/03_find_shifters.py
```

---

## Step 6 — COG functional enrichment (external script)

**Steps 6–8 are supplementary exploratory analysis, not part of the
published manuscript** (which reports KEGG pathway enrichment among the
non-growing gene-level profiles, via `analysis/kegg_enrichment_s2.py`, not COG
categories — see Supplementary Data S2). They still depend on the legacy
four-model parametric workflow in `analysis/batch_fit.jl`; neither that fit nor
its fitted parameters are inputs to the published results.

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

## Extra / exploratory material (not used in the manuscript)

Beyond Steps 5–8, the repository carries a few further scripts and committed
output directories that no manuscript figure, table or supplementary dataset
depends on. They are kept for transparency about what was explored; nothing in
Steps 1–4 reads them.

| Path | What it is |
|---|---|
| `analysis/ml_keio.py` | Random forest (with a Spearman-correlation fallback) predicting `gr` / `N_max` / `lag` from COG functional category, run locally with scikit-learn. Superseded by the GUIbiont-API version below. |
| `analysis/run_cog_ml_via_guibiont.py` | API-driven rerun of the same COG machine-learning screen through GUIbiont's `/api/ml-downstream` route, so every number comes from the same DecisionTree.jl forest the UI exposes. |
| `analysis/enrichment_nongrowers.py` | Auxotroph enrichment of the M63 non-growing pre-screen class (Fisher exact against amino-acid-biosynthesis, wider KEGG biosynthesis, and COG E gene sets). Exploratory only — the manuscript's non-grower enrichment is the KEGG analysis in Step 3b. |
| `figures/figures.jl` | Static publication figures. Run from the repo root with `julia --project=figures figures/figures.jl` (the CairoMakie dependency lives in `figures/Project.toml`, not `analysis/Project.toml`). |
| `results/enrichment/` | Committed output of `analysis/enrichment.py` (Step 6): COG enrichment CSV, gene→COG assignments, and the enrichment figure. |
| `results/ml_keio/` | Committed output of `analysis/ml_keio.py`: per-parameter COG feature-importance CSVs and figures. |
| `results/enrichment_97_nongrowers.csv` | Committed output of `analysis/enrichment_nongrowers.py`. |

---

## Summary of manuscript-reproduction outputs

The following tracked files are produced by the required manuscript-reproduction
steps (Steps 1--4, including Step 3b):

| File | How |
|---|---|
| `results/keio_lb_gene_means.csv` | `scripts/01_build_gene_means.py` (Step 1) |
| `results/keio_m63_gene_means.csv` | `scripts/01_build_gene_means.py` (Step 1) |
| `docs/data/curves_data.json` | `analysis/analyse.jl` (Step 3) |
| `results/keio_s1_metadata.csv` | `analysis/analyse.jl` (Step 3) |
| `results/keio_m63_nongrowing_genes.json` | `analysis/analyse.jl` (Step 3) |
| `results/keio_kegg_pathway_memberships.csv` | `analysis/kegg_enrichment_s2.py` (Step 3b) |
| `results/keio_nongrowing_enrichment.csv` | `analysis/kegg_enrichment_s2.py` (Step 3b) |
| `results/keio_loglin_results.csv` | `analysis/run_keio_loglin_via_guibiont.py` (Step 4) |

Step 2 creates registered experiment directories under the external GUIbiont
`Clean_data/` directory. The manual GUI exports and the outputs of Steps 5--8
are exploratory, are not required for the manuscript, and are documented in
their respective sections above rather than in this summary.
