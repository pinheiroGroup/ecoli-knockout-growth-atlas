from __future__ import annotations
"""Auxotroph enrichment of the M63 non-growing pre-screen class.

Tests the constant-curve pre-screen non-growers against three gene sets with a
one-sided Fisher exact test:

  1. amino-acid biosynthesis  -- gene name prefixes (arg..gly)
  2. wider biosynthesis       -- KEGG E. coli biosynthesis pathways
                                 (amino acids eco01230, cofactors eco01240,
                                  purine eco00230, pyrimidine eco00240)
  3. COG E                    -- amino-acid transport & metabolism

The wider set is fetched live from the KEGG REST API so the gene list is
reproducible rather than hand-curated. Writes enrichment_<n>_nongrowers.csv,
where <n> is the size of the pre-screen class in curves_data.json.
"""
import csv
import json
import urllib.request
from pathlib import Path

from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CURVES = ROOT / "docs" / "data" / "curves_data.json"
COG_CSV = ROOT / "results" / "enrichment" / "gene_cog_assignments.csv"

AA_PREFIXES = ("arg", "aro", "cys", "his", "ilv", "leu", "lys",
               "met", "phe", "pro", "ser", "thr", "trp", "tyr", "gly")
KEGG_BIOSYNTH_PATHWAYS = ("eco01230", "eco01240", "eco00230", "eco00240")


def kegg(url: str) -> str:
    return urllib.request.urlopen(url, timeout=30).read().decode()


def kegg_bnum_to_name() -> dict[str, str]:
    out = {}
    for line in kegg("https://rest.kegg.jp/list/eco").splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[0].startswith("eco:"):
            name = parts[3].split(";")[0].strip().lower()
            if name:
                out[parts[0].split(":")[1]] = name
    return out


def kegg_pathway_genes(pathway: str, bnum_name: dict[str, str]) -> set[str]:
    names = set()
    for line in kegg(f"https://rest.kegg.jp/link/eco/{pathway}").splitlines():
        if "\t" in line:
            b = line.split("\t")[1].split(":")[1]
            if b in bnum_name:
                names.add(bnum_name[b])
    return names


def fisher_row(label: str, pool: set[str], nongrowers: set[str],
               universe: set[str]) -> list:
    pool = pool & universe
    n_total = len(universe)
    n_ng = len(nongrowers)
    a = len(pool & nongrowers)
    size = len(pool)
    b, c = size - a, n_ng - a
    d = n_total - size - c
    odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    expected = n_ng * size / n_total
    fold = a / expected if expected else 0.0
    return [label, size, a, round(expected, 2), round(fold, 2), round(odds, 2), p]


def main() -> None:
    data = json.loads(CURVES.read_text())
    universe = {g["gene"].lower() for g in data["genes"]}
    nongrowers = {g.lower() for g in data["nongrowing_genes"]["M63"]}
    out_name = f"enrichment_{len(nongrowers)}_nongrowers.csv"

    aa_set = {g for g in universe if g[:3] in AA_PREFIXES}

    bnum_name = kegg_bnum_to_name()
    wider_set = set()
    for pw in KEGG_BIOSYNTH_PATHWAYS:
        wider_set |= kegg_pathway_genes(pw, bnum_name)

    cog_rows = list(csv.DictReader(COG_CSV.open()))
    cog_e = {r["gene"].lower() for r in cog_rows if r["primary_cog"] == "E"}

    rows = [
        ["gene_set", "set_size", "in_nongrowers", "expected",
         "fold_enrichment", "odds_ratio", "p_fisher_greater"],
        fisher_row("amino-acid biosynthesis (name prefixes)",
                   aa_set, nongrowers, universe),
        fisher_row("wider biosynthesis (KEGG AA + cofactor + nucleotide)",
                   wider_set, nongrowers, universe),
        fisher_row("COG E (amino-acid transport & metabolism)",
                   cog_e, nongrowers, universe),
    ]

    for base in (ROOT / "results",):
        with (base / out_name).open("w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        print("wrote", base / out_name)
    for r in rows[1:]:
        print(r)


if __name__ == "__main__":
    main()
