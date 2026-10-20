# Corpus attribution

The evaluation corpus is the set of **NTSB Marine Investigation Reports published in 2025 and 2026** (`MIR-25-01` to `MIR-25-47`, `MIR-26-01` to `MIR-26-30`). They are listed with links in [`sources.yaml`](sources.yaml) and in the README.

Courtesy: National Transportation Safety Board.

## Status

From the NTSB website policy (https://www.ntsb.gov/about/Policies/Pages/Policies.aspx), "Text Content":

> Text appearing on NTSB Web pages, in reports, recommendation, and public dockets, unless otherwise noted, was prepared by employees of the United States Government as part of their official duties and, therefore, is not subject to copyright. … By publishing its reports and recommendations and opening public dockets, the NTSB has placed the contents prepared by its staff in the public domain.

The NTSB logo and seal are not covered by this statement. Any material in a report that the NTSB marks as third-party copyrighted remains under its owner's copyright. This project uses the report text for evaluation and does not rely on such material.

## Not redistributed here

The PDFs are not committed to this repository. `uv run rag-eval fetch-corpus` downloads them from ntsb.gov and verifies each file against the SHA-256 pinned in `sources.yaml`. See [ADR 0003](../docs/decisions/0003-corpus-choice.md) for why this corpus was chosen.

The MIT license of this repository covers the code and documentation, not the corpus.
