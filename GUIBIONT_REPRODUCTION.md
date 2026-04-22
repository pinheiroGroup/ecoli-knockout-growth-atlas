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

- GUIbiont server running (`julia --project=. web_server.jl` from the GUIbiont repo)
- Python ≥ 3.9 with `pandas`, `numpy`, `openpyxl`
- Install dependencies once: `pip install pandas numpy openpyxl`

---

## Step 1 — Build per-gene mean curves

The raw data contains individual replicate curves keyed by curve ID.
This script reads the metadata mapping, aggregates replicates by gene,
and outputs one CSV per medium ready to load into GUIbiont.

Save as `scripts/01_build_gene_means.py` and run from the repo root:

```python
#!/usr/bin/env python3
"""Build per-gene mean curves for LB and M63 from the Keio atlas raw data."""
import numpy as np
import pandas as pd

DATA_DIR    = "data"
RESULTS_DIR = "results"

# --- Load metadata: curve_id → (gene_name, medium) ---
meta = pd.read_excel(f"{DATA_DIR}/Curves_knockouts_media.xlsx", header=0)
# Expected columns: curve_id, jw_id, gene_name, gene_category, medium
meta.columns = [c.strip() for c in meta.columns]
meta["curve_id"]   = meta.iloc[:, 0].astype(str).str.strip()
meta["gene_name"]  = meta.iloc[:, 2].astype(str).str.strip()
meta["medium"]     = meta.iloc[:, 4].astype(str).str.strip()

# --- Load all curves (row index + Time column + Curve##### columns) ---
# all_curves.csv has two halves: rows 0–199 (LB, valid time) and rows 200–399
# (M63, NaN time). Both media share the same 200-point plate reader schedule
# (0.25–50 h in 0.25 h steps), so we reuse the LB time axis for M63.
curves = pd.read_csv(f"{DATA_DIR}/all_curves.csv", index_col=0)
time_col = curves.columns[0]
all_times = curves[time_col].values.astype(float)

valid_time_mask = ~np.isnan(all_times)
shared_times    = all_times[valid_time_mask]   # 200 time points, 0.25–50 h

# Row ranges for each medium in the combined CSV
MEDIUM_ROWS = {"LB": slice(0, 200), "M63": slice(200, 400)}

for medium in ["LB", "M63"]:
    subset   = meta[meta["medium"] == medium]
    gene_map = dict(zip(subset["curve_id"], subset["gene_name"]))
    row_slice = MEDIUM_ROWS[medium]

    groups: dict[str, list] = {}
    for curve_id, gene in gene_map.items():
        col = f"Curve{curve_id}"
        if col not in curves.columns:
            col = curve_id
        if col not in curves.columns:
            continue
        series = curves[col].values.astype(float)[row_slice]
        # Mask trailing zeros: zeros after the last OD > 0.01 are artefacts
        # (blank Excel cells read as 0, or evaporated/failed wells)
        valid_idx = np.where(series > 0.01)[0]
        if valid_idx.size > 0:
            series[valid_idx[-1] + 1:] = np.nan
        groups.setdefault(gene, []).append(series)

    result = {"Time": shared_times}
    for gene, reps in groups.items():
        mat = np.vstack(reps)           # shape: n_reps × 200
        result[gene] = np.nanmean(mat, axis=0)

    out = pd.DataFrame(result)
    out.to_csv(f"{RESULTS_DIR}/keio_{medium.lower()}_gene_means.csv", index=False)
    print(f"{medium}: {len(result) - 1} genes written")
```

```bash
python scripts/01_build_gene_means.py
```

Output: `results/keio_lb_gene_means.csv`, `results/keio_m63_gene_means.csv`  
Format: `Time, geneA, geneB, ...` — one column per gene, rows = time points.

---

## Step 2 — Register as GUIbiont experiments

GUIbiont's Batch Fit tab reads from `$GUIBIONT/Clean_data/<experiment>/`.
This script creates the two experiment directories with the correct file names.

Save as `scripts/02_register_experiments.py` and run from the repo root:

```python
#!/usr/bin/env python3
"""Register Keio gene-mean CSVs as GUIbiont experiments."""
import os
import pandas as pd

GUIBIONT   = "/path/to/GUIbiont"           # ← change this
RESULTS    = "results"

for medium in ["lb", "m63"]:
    exp_dir = os.path.join(GUIBIONT, "Clean_data", f"keio_{medium}")
    os.makedirs(exp_dir, exist_ok=True)

    src = os.path.join(RESULTS, f"keio_{medium}_gene_means.csv")
    dst = os.path.join(exp_dir, "data_channel_1.csv")

    import shutil
    shutil.copy(src, dst)

    # Minimal annotation: well=gene, condition=WT, no blanks
    df   = pd.read_csv(src)
    genes = df.columns[1:]   # skip Time
    rows  = [[g, "WT", "", "", ""] for g in genes]
    pd.DataFrame(rows).to_csv(
        os.path.join(exp_dir, "annotation_clean.csv"),
        header=False, index=False
    )
    print(f"Registered: keio_{medium}  ({len(genes)} wells)")
```

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
   (expect ~3–5 min for ~3,800 genes; convergence rate should be ≥ 98%)
6. Click **Download CSV** → save as `results/keio_batch_fit_lb.csv`

Repeat for `keio_m63`, saving as `results/keio_batch_fit_m63.csv`.

---

## Step 5 — Identify cluster shifters (external script)

Genes whose cluster assignment differs between LB and M63:

```python
#!/usr/bin/env python3
import pandas as pd

lb  = pd.read_csv("results/clusters_lb.csv") \
        .rename(columns={"Cluster": "cluster_lb",  "Well": "gene"})[["gene","cluster_lb"]]
m63 = pd.read_csv("results/clusters_m63.csv") \
        .rename(columns={"Cluster": "cluster_m63", "Well": "gene"})[["gene","cluster_m63"]]

joined   = lb.merge(m63, on="gene")
shifters = joined[joined["cluster_lb"] != joined["cluster_m63"]]

joined.to_csv("results/cluster_assignments_both.csv",   index=False)
shifters.to_csv("results/cluster_shifters.csv",          index=False)
print(f"Total genes: {len(joined)}")
print(f"Shifters:    {len(shifters)}  ({100*len(shifters)/len(joined):.1f}%)")
```

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

Save as `scripts/04_build_cog_matrix.py`:

```python
#!/usr/bin/env python3
"""Build a gene × COG-category multi-hot feature matrix."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from ml_keio import build_gene_cog_map
import pandas as pd

df    = pd.read_csv("results/keio_batch_fit_lb.csv")
genes = df["gene"].dropna().unique().tolist()

gene_cog = build_gene_cog_map(genes)
all_cats = sorted({c for cats in gene_cog.values() for c in cats})

rows = [
    {"gene": g, **{f"COG_{c}": int(c in gene_cog.get(g.lower(), [])) for c in all_cats}}
    for g in genes
]
pd.DataFrame(rows).to_csv("results/cog_feature_matrix.csv", index=False)
print(f"Saved: {len(rows)} genes × {len(all_cats)} COG categories")
```

```bash
python scripts/04_build_cog_matrix.py
```

Also prepare cleaned per-medium fit CSVs (filter converged, cap lag):

```python
#!/usr/bin/env python3
"""Clean batch fit results for ML Analysis upload."""
import numpy as np, pandas as pd

for medium in ["lb", "m63"]:
    df = pd.read_csv(f"results/keio_batch_fit_{medium}.csv")
    df = df[df["converged"].astype(str).str.lower() == "true"].copy()
    df.loc[df["lag"] >= 49.9, "lag"] = np.nan
    df.to_csv(f"results/keio_batch_fit_{medium}_clean.csv", index=False)
    print(f"{medium.upper()}: {len(df)} converged rows")
```

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
