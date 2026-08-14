#!/usr/bin/env python3
"""Drive the full Keio log-linear batch fit through GUIbiont's
/api/batch-fit-loglin endpoint and write the canonical per-gene result CSV.

Steps:
  1. POST /api/batch-fit-loglin for each medium (keio_lb, keio_m63).
  2. Poll /api/batch-fit/progress/{job_id} until done.
  3. Collect results and write results/keio_loglin_results.csv with the
     schema (gene, medium, gr_loglin, gr_loglin_se, gr_max_sliding,
     t_exp_start, t_exp_end, doubling_time, R_squared, lag_loglin,
     N_max_emp, converged).
This is the actual GUIbiont workflow a user would run from the Batch Fit
tab with "Log-linear only" selected.

Run:
    /usr/bin/python analysis/run_keio_loglin_via_guibiont.py
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
import json
from urllib.error import HTTPError, URLError

import pandas as pd
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(REPO, "results")
OUT_CSV = os.path.join(RES, "keio_loglin_results.csv")

API = os.environ.get("GUIBIONT_API", "http://localhost:8080")
MEDIA = ("keio_lb", "keio_m63")

# Parameters reported for the Keio analysis in the GUIbiont manuscript.
LOGLIN_PARAMS = {
    "blank_subtraction":       False,
    "type_of_smoothing":       "rolling_avg",
    "pt_avg":                  5,
    "pt_smoothing_derivative": 5,
    "pt_min_size_of_win":      5,
    "type_of_win":             "maximum",
    "threshold_of_exp":        0.9,
    "skip_flat_threshold":     0.0,   # zero keeps all wells
}


def _post(path, body, timeout=120):
    req = urllib.request.Request(
        f"{API}{path}", method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode(),
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def _get(path, timeout=120):
    req = urllib.request.Request(f"{API}{path}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def submit_and_wait(experiment, poll_interval=5.0, max_wait=2400.0):
    """Submit a batch-fit-loglin job and poll until done."""
    payload = {"experiment": experiment, **LOGLIN_PARAMS}
    print(f"[{experiment}] POST /api/batch-fit-loglin "
          f"(this kicks off all {3885} wells; expect a few minutes)...")
    status, body = _post("/api/batch-fit-loglin", payload)
    if status != 200:
        raise RuntimeError(f"submission failed: HTTP {status}  body={body}")

    job_id = body["job_id"]
    total  = body["total"]
    print(f"[{experiment}] job_id = {job_id},  total = {total}")

    start = time.time()
    last_print = 0
    while True:
        s, p = _get(f"/api/batch-fit/progress/{job_id}")
        if s != 200:
            raise RuntimeError(f"progress failed: HTTP {s}")
        completed = int(p.get("completed", 0))
        if (time.time() - last_print) > 8:
            cw = p.get("current_well", "")
            print(f"[{experiment}] {completed}/{total}  current={cw!r}  "
                  f"elapsed={time.time()-start:.0f}s")
            last_print = time.time()
        if p.get("status") == "done":
            print(f"[{experiment}] done in {time.time()-start:.1f}s, "
                  f"summary={p['summary']}")
            return p
        if time.time() - start > max_wait:
            raise TimeoutError(
                f"[{experiment}] job did not finish in {max_wait}s "
                f"({completed}/{total})")
        time.sleep(poll_interval)


def results_to_rows(job_payload, medium_label):
    """Flatten GUIbiont's results list into the same schema as
    results/keio_loglin_results.csv, the CSV this script writes.

    Carries the two model-free companions returned with the log-linear
    estimator: lag_loglin (Buchanan tangent-intercept lag) and N_max_emp
    (stationary-region mean OD)."""
    rows = []
    for r in job_payload["results"]:
        rows.append({
            "gene":            r["well"],
            "medium":          medium_label,
            "gr_loglin":       r.get("gr_loglin"),
            "gr_loglin_se":    r.get("gr_loglin_se"),
            "gr_max_sliding":  r.get("gr_max_sliding"),
            "t_exp_start":     r.get("t_exp_start_loglin"),
            "t_exp_end":       r.get("t_exp_end_loglin"),
            "doubling_time":   r.get("doubling_time_loglin"),
            "R_squared":       r.get("R_squared_loglin"),
            "lag_loglin":      r.get("lag_loglin"),
            "N_max_emp":       r.get("N_max_emp"),
            "converged":       bool(r.get("loglin_converged", False)),
        })
    # Errors / skipped wells: still emit a row with NaN so the join is total.
    errors = job_payload.get("errors") or job_payload.get("summary", {}).get("errors", [])
    for err in errors:
        # Errors look like "Well 'xxx': reason"
        gene = err.split("'", 2)[1] if "'" in err else "<unknown>"
        rows.append({"gene": gene, "medium": medium_label,
                     "gr_loglin": np.nan, "gr_loglin_se": np.nan,
                     "gr_max_sliding": np.nan, "t_exp_start": np.nan,
                     "t_exp_end": np.nan, "doubling_time": np.nan,
                     "R_squared": np.nan,
                     "lag_loglin": np.nan, "N_max_emp": np.nan,
                     "converged": False})
    return rows


def main():
    # Sanity: server is live and our new endpoint is there.
    try:
        _, exps = _get("/api/experiments")
    except (HTTPError, URLError) as e:
        print(f"ERROR: cannot reach GUIbiont at {API}: {e}", file=sys.stderr)
        sys.exit(1)
    for m in MEDIA:
        if m not in exps:
            print(f"ERROR: experiment {m!r} not loaded on server", file=sys.stderr)
            sys.exit(2)

    all_rows = []
    media_label = {"keio_lb": "LB", "keio_m63": "M63"}
    for exp in MEDIA:
        payload = submit_and_wait(exp)
        all_rows.extend(results_to_rows(payload, media_label[exp]))

    df = (pd.DataFrame(all_rows)
          .sort_values(["medium", "gene"], kind="stable")
          .reset_index(drop=True))
    if len(df) != 7770 or df.duplicated(["gene", "medium"]).any():
        raise RuntimeError("expected 7,770 unique gene-medium rows")
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(df)} rows)")

    print("\nMissing values in canonical output:")
    for column in ("gr_loglin", "lag_loglin", "R_squared", "N_max_emp"):
        print(f"  {column}: {int(df[column].isna().sum())}")

    # Per-medium quantile summary
    print("\nPer-medium mu_max quantiles (via GUIbiont):")
    for med in ("LB", "M63"):
        sub = df[(df["medium"] == med) & df["converged"]]["gr_loglin"]
        if not sub.empty:
            print(f"  {med}: n={len(sub)}  "
                  f"p05={sub.quantile(0.05):.3f}  "
                  f"p50={sub.median():.3f}  "
                  f"p95={sub.quantile(0.95):.3f}")


if __name__ == "__main__":
    main()
