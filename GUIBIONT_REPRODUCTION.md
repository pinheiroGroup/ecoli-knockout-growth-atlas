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

## Step 3 — Clustering (GUIbiont interface)

Do this **twice** — once for LB, once for M63.

### 3a — Cluster sweep (find optimal k)

1. Open the **Clustering** tab → click **File** mode
2. Upload `results/keio_lb_gene_means.csv`
3. Open **Advanced options**:
   - Smooth method: `lowess`, frac `0.05`
   - Check **Interpolate to common grid**
     - Grid points: `100`
     - t_start quantile: `0.05`
     - t_end quantile: `0.95`
   - Cluster method: `kmeans`
4. Click **Sweep** (k = 2 to 10)
5. Look at the **WCSS elbow plot** — note the suggested k (expect ~4 for LB, ~4 for M63, where one cluster is the non-growing sentinel)

### 3b — Run clustering at optimal k

1. Set **k** to the elbow value from the sweep
2. Click **Run**
3. Inspect the cluster grid — one cluster should contain the non-growing/flat genes
4. Click **Export all (CSV)** → save as `results/clusters_lb.csv`

Repeat steps 3a–3b for M63, saving as `results/clusters_m63.csv`.

---

## Step 4 — Batch fitting (GUIbiont interface)

Do this **twice** — once for LB, once for M63.

1. Open the **Batch Fit** tab
2. Select experiment `keio_lb`
3. **Models**: check all four — `logistic`, `gompertz`, `baranyi_richards`, `aHPM`
4. Leave AICc model selection enabled (default)
5. Click **Run**  
   (runtime depends heavily on hardware and Julia thread count — convergence rate should be ≥ 98%)
6. Click **Download CSV** → save as `results/keio_batch_fit_lb.csv`

Repeat for `keio_m63`, saving as `results/keio_batch_fit_m63.csv`.

---

## Step 5 — Identify cluster shifters (external script)

Genes whose cluster assignment differs between LB and M63:

```bash
python scripts/03_find_shifters.py
```

---

## Step 6 — COG functional enrichment (external script)

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
2. **Fit results CSV**: upload `results/keio_batch_fit_lb_clean.csv`
3. **Label column**: `gene`
4. **Feature matrix CSV**: upload `results/cog_feature_matrix.csv`
5. **Parameters to analyse**: select `gr`, `N_max`, `lag`
6. Click **Run**

The interface shows:
- **Spearman correlations** — ranked bar chart per parameter
- **Random forest feature importance** — top COG categories per parameter
- **Partial dependence plots** — marginal effect of top 5 COG categories

Repeat with `keio_batch_fit_m63_clean.csv` for M63.

---

## Summary of files produced

| File | How |
|---|---|
| `results/keio_lb_gene_means.csv` | Script 01 |
| `results/keio_m63_gene_means.csv` | Script 01 |
| `results/clusters_lb.csv` | GUIbiont Clustering tab |
| `results/clusters_m63.csv` | GUIbiont Clustering tab |
| `results/keio_batch_fit_lb.csv` | GUIbiont Batch Fit tab |
| `results/keio_batch_fit_m63.csv` | GUIbiont Batch Fit tab |
| `results/cluster_assignments_both.csv` | Script 03 |
| `results/cluster_shifters.csv` | Script 03 |
| `results/enrichment/` | `enrichment.py` |
| `results/cog_feature_matrix.csv` | Script 04 |
| `results/keio_batch_fit_{lb,m63}_clean.csv` | Script 05 |
