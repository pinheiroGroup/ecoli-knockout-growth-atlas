#!/usr/bin/env python3
"""Register Keio gene-mean CSVs as GUIbiont experiments."""
import os
import pandas as pd

GUIBIONT   = "../GUIbiont-repo" 
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

