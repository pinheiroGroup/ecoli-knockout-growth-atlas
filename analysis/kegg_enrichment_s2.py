#!/usr/bin/env python3
"""KEGG pathway membership and enrichment for the M63 non-growing strains.

Produces the two files published as Supplementary Data S2:

    results/keio_kegg_pathway_memberships.csv
    results/keio_nongrowing_enrichment.csv

Two gene sets are tested, both restricted to genes represented in the Keio
screen, exactly as the manuscript describes them:

    amino-acid biosynthesis            eco01230
    wider biosynthesis-related set     eco01230 + eco01240 + eco00230 + eco00240

This supersedes the amino-acid row in the older enrichment_97_nongrowers.csv,
which selected amino-acid genes by name prefix (arg*, aro*, ...) rather than by
KEGG membership and therefore reported a different set size and hit count.

The non-growing class is *not* recomputed here. It comes from GUIbiont's
clustering pre-screen, run on the full-length replicate-averaged M63 matrix,
and is supplied as a JSON list so this script cannot drift from the classifier
the paper describes.

Run:
    python analysis/kegg_enrichment_s2.py --non-growers <nongrowers.json>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
CACHE = RESULTS / "cache"

AA_PATHWAY = "eco01230"
WIDER_PATHWAYS = ("eco01230", "eco01240", "eco00230", "eco00240")


def kegg_get(url: str, cache_name: str) -> str:
    """Fetch a KEGG endpoint, caching the response so reruns are offline."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / cache_name
    if cached.exists():
        return cached.read_text()
    text = urllib.request.urlopen(url, timeout=60).read().decode()
    cached.write_text(text)
    return text


def bnum_to_gene_name() -> dict[str, str]:
    """Map b-numbers to primary gene names, lowercased for matching."""
    out = {}
    for line in kegg_get("https://rest.kegg.jp/list/eco", "kegg_list_eco.tsv").splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[0].startswith("eco:"):
            name = parts[3].split(";")[0].strip().lower()
            if name:
                out[parts[0].split(":")[1]] = name
    return out


def pathway_gene_names(pathway: str, bnum: dict[str, str]) -> set[str]:
    names = set()
    text = kegg_get(f"https://rest.kegg.jp/link/eco/{pathway}",
                    f"kegg_link_{pathway}.tsv")
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].startswith("eco:"):
            b = parts[1].split(":")[1]
            if b in bnum:
                names.add(bnum[b])
    return names


def enrichment(screen: set[str], non_growers: set[str],
               gene_set: set[str]) -> dict:
    """One-sided Fisher exact test for over-representation in the class.

    The 2x2 table is built on the whole screen, so the comparison is against
    the rest of the collection rather than against a background of all E. coli
    genes.
    """
    in_set_in_class = len(non_growers & gene_set)
    in_set_not_class = len(gene_set - non_growers)
    not_set_in_class = len(non_growers - gene_set)
    not_set_not_class = len(screen - non_growers - gene_set)

    table = [[in_set_in_class, in_set_not_class],
             [not_set_in_class, not_set_not_class]]
    odds_ratio, p_value = fisher_exact(table, alternative="greater")

    expected = len(gene_set) * len(non_growers) / len(screen)
    fold = in_set_in_class / expected if expected else float("nan")
    return {
        "set_size_in_screen": len(gene_set),
        "class_size": len(non_growers),
        "screen_size": len(screen),
        "observed_in_class": in_set_in_class,
        "in_set_not_in_class": in_set_not_class,
        "in_class_not_in_set": not_set_in_class,
        "neither": not_set_not_class,
        "expected_in_class": round(expected, 4),
        "fold_enrichment": round(fold, 4),
        "odds_ratio": round(odds_ratio, 4),
        "p_fisher_one_sided": p_value,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--non-growers", required=True,
                    help="JSON with a 'genes' list from the GUIbiont pre-screen")
    ap.add_argument("--screen", default=str(RESULTS / "keio_m63_gene_means.csv"),
                    help="matrix whose header lists every gene in the screen")
    args = ap.parse_args()

    non_growers = {g.lower() for g in json.loads(
        Path(args.non_growers).read_text())["genes"]}
    with open(args.screen, newline="") as fh:
        header = next(csv.reader(fh))
    screen = {g.lower() for g in header[1:] if g}

    missing = non_growers - screen
    if missing:
        print(f"non-growers absent from the screen: {sorted(missing)[:5]}",
              file=sys.stderr)
        return 1

    bnum = bnum_to_gene_name()
    aa_set = pathway_gene_names(AA_PATHWAY, bnum) & screen
    wider_set = set()
    per_pathway = {}
    for pathway in WIDER_PATHWAYS:
        genes = pathway_gene_names(pathway, bnum) & screen
        per_pathway[pathway] = genes
        wider_set |= genes

    # Per-gene memberships for every gene in the screen.
    rows = []
    for gene in sorted(screen):
        rows.append({
            "gene": gene,
            "in_screen": True,
            "non_growing_m63": gene in non_growers,
            "amino_acid_biosynthesis_eco01230": gene in aa_set,
            "wider_biosynthesis_set": gene in wider_set,
            **{f"in_{p}": gene in per_pathway[p] for p in WIDER_PATHWAYS},
        })
    memberships = RESULTS / "keio_kegg_pathway_memberships.csv"
    with open(memberships, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    results = [
        {"gene_set": "amino-acid biosynthesis (KEGG eco01230)",
         **enrichment(screen, non_growers, aa_set)},
        {"gene_set": "wider biosynthesis (eco01230/eco01240/eco00230/eco00240)",
         **enrichment(screen, non_growers, wider_set)},
    ]
    out = RESULTS / "keio_nongrowing_enrichment.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    for r in results:
        print(f"{r['gene_set']}\n"
              f"  set size in screen : {r['set_size_in_screen']}\n"
              f"  observed in class  : {r['observed_in_class']} of {r['class_size']}\n"
              f"  expected           : {r['expected_in_class']}\n"
              f"  fold enrichment    : {r['fold_enrichment']}\n"
              f"  odds ratio         : {r['odds_ratio']}\n"
              f"  P (one-sided)      : {r['p_fisher_one_sided']:.3g}")
    print(f"\nWrote {memberships}\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
