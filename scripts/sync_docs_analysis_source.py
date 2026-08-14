#!/usr/bin/env python3
"""Synchronize the source-code tab with analysis/analyse.jl.

The GitHub Pages site embeds the complete analysis script for offline browsing.
Keeping this as a generated block prevents the displayed setup and parameters
from drifting from the executable source.
"""

from __future__ import annotations

import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis" / "analyse.jl"
PAGE = ROOT / "docs" / "index.html"
BEGIN = "<!-- BEGIN GENERATED ANALYSIS SOURCE -->"
END = "<!-- END GENERATED ANALYSIS SOURCE -->"


def main() -> None:
    page = PAGE.read_text(encoding="utf-8")
    if page.count(BEGIN) != 1 or page.count(END) != 1:
        raise SystemExit("expected exactly one generated source block")

    source = SOURCE.read_text(encoding="utf-8").replace("\r\n", "\n")
    generated = (
        f"{BEGIN}\n"
        f'<pre><code class="language-julia">{html.escape(source, quote=False)}'
        f"</code></pre>\n"
        f"{END}"
    )
    prefix, rest = page.split(BEGIN, 1)
    _, suffix = rest.split(END, 1)
    updated = (prefix + generated + suffix).replace("\r\n", "\n")
    PAGE.write_text(updated, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
