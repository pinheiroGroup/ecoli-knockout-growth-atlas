#!/usr/bin/env python3
"""Drive the full Keio log-linear batch fit through GUIbiont's new
/api/batch-fit-loglin endpoint and write the per-gene μ_max CSV.

Steps:
  1. POST /api/batch-fit-loglin for each medium (keio_lb, keio_m63).
  2. Poll /api/batch-fit/progress/{job_id} until done.
  3. Collect results and write results/keio_loglin_via_guibiont.csv with
     the same schema as keio_loglin_results.csv (gene, medium, gr_loglin,
     gr_loglin_se, gr_max_sliding, t_exp_start, t_exp_end, doubling_time,
     R_squared, converged).
  4. Cross-check against the existing offline results — they must match
     to the 5 % tolerance we already verified at the unit level
     (companion path uses parametric floor 0.01, batch-fit-loglin uses
     log-lin floor 1e-4; both agree for healthy curves but can differ
     in the right tail).

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
OUT_CSV   = os.path.join(RES, "keio_loglin_via_guibiont.csv")
REF_CSV   = os.path.join(RES, "keio_loglin_results.csv")

API = os.environ.get("GUIBIONT_API", "http://localhost:9090")
MEDIA = ("keio_lb", "keio_m63")

# Match the parameters my offline batch_fit_loglin.jl used so we can
# cross-check against the existing reference CSV.
LOGLIN_PARAMS = {
    "pt_avg":                  5,
    "pt_smoothing_derivative": 5,
    "pt_min_size_of_win":      5,
    "threshold_of_exp":        0.9,
    "skip_flat_threshold":     0.0,   # 0 → don't skip flats (we want all wells)
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
          f"(this kicks off all {3885} wells; expect a few minutes)…")
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
    keio_loglin_results.csv produced by analysis/batch_fit_loglin.jl.

    Carries the two model-free companions Kinbiont now appends to the
    log-linear estimator: lag_loglin (Buchanan tangent-intercept lag) and
    N_max_emp (95th percentile of smoothed OD). NaN when the upgraded
    Kinbiont/GUIbiont is not running."""
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

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(df)} rows)")

    # ── Cross-check against the offline reference ─────────────────────────
    if not os.path.exists(REF_CSV):
        print("(no offline reference to compare against — done)")
        return
    ref = pd.read_csv(REF_CSV)
    merged = df.merge(
        ref[["gene", "medium", "gr_loglin"]].rename(columns={"gr_loglin": "gr_ref"}),
        on=["gene", "medium"], how="inner",
    )
    merged["delta"] = (merged["gr_loglin"] - merged["gr_ref"]).abs()
    finite = merged.dropna(subset=["gr_loglin", "gr_ref"])
    print(f"\n── Cross-check vs {os.path.basename(REF_CSV)} "
          f"({len(finite)} comparable pairs) ──")
    print(f"  max |Δ|:    {finite['delta'].max():.6g}")
    print(f"  mean |Δ|:   {finite['delta'].mean():.6g}")
    print(f"  median |Δ|: {finite['delta'].median():.6g}")
    print(f"  # exact matches: {(finite['delta'] == 0).sum()} / {len(finite)}")
    n_big = int((finite["delta"] > 0.05).sum())
    print(f"  # |Δ| > 0.05 (5% physiological μ_max units): {n_big} / {len(finite)}")

    # Per-medium quantile summary
    print("\nPer-medium μ_max quantiles (via GUIbiont):")
    for med in ("LB", "M63"):
        sub = df[(df["medium"] == med) & df["converged"]]["gr_loglin"]
        if not sub.empty:
            print(f"  {med}: n={len(sub)}  "
                  f"p05={sub.quantile(0.05):.3f}  "
                  f"p50={sub.median():.3f}  "
                  f"p95={sub.quantile(0.95):.3f}")


if __name__ == "__main__":
    main()
