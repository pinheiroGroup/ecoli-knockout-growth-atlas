#!/usr/bin/env python3
"""Register Keio gene-mean CSVs as GUIbiont experiments."""
import os
import sys
import shutil
import pandas as pd

RESULTS = "results"


def pick_guibiont_dir() -> str:
    """Resolve the GUIbiont repo path from CLI arg, env var, GUI picker, or stdin."""
    if len(sys.argv) > 1:
        return sys.argv[1]

    env_dir = os.environ.get("GUIBIONT_DIR")
    if env_dir:
        return env_dir

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        path = input("Path to your GUIbiont repository folder: ").strip()
        return path

    try:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="Select your GUIbiont repository folder")
        root.destroy()
        return path
    except tk.TclError:
        # No display available (headless / SSH / CI).
        path = input("Path to your GUIbiont repository folder: ").strip()
        return path


GUIBIONT = pick_guibiont_dir()

if not GUIBIONT:
    raise ValueError("No folder selected. Please select your GUIbiont repository.")

GUIBIONT = os.path.abspath(os.path.expanduser(GUIBIONT))

if not os.path.isdir(GUIBIONT):
    raise ValueError(f"Not a directory: {GUIBIONT}")

for medium in ["lb", "m63"]:
    exp_dir = os.path.join(GUIBIONT, "Clean_data", f"keio_{medium}")
    os.makedirs(exp_dir, exist_ok=True)

    src = os.path.join(RESULTS, f"keio_{medium}_gene_means.csv")
    dst = os.path.join(exp_dir, "data_channel_1.csv")

    shutil.copy(src, dst)

    # Minimal annotation: well=gene, condition=WT, no blanks
    df = pd.read_csv(src)
    genes = df.columns[1:]   # skip Time
    rows = [[g, "WT", "", "", ""] for g in genes]

    pd.DataFrame(rows).to_csv(
        os.path.join(exp_dir, "annotation_clean.csv"),
        header=False,
        index=False
    )

    print(f"Registered: keio_{medium} ({len(genes)} wells)")
