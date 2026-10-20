# 0003: Corpus choice

- **Status:** Accepted (option A)
- **Date:** 2026-10-21

## Requirements

1. **Permissive for redistribution:** the corpus is committed to this MIT-licensed repository with attribution.
2. **New or niche enough to limit memorisation** by the system under test, `qwen3:8b`. Qwen3 was released in April 2025 and its training cutoff is not published, so we treat anything published from 2025 on as unlikely to be memorised. The closed-book baseline measures this per question regardless.
3. **Deep enough for multi-document questions:** comparisons and facts spread across documents, plus enough material for ~80–100 approved questions with ~15–20% unanswerable.

## Candidates

### A. NTSB marine investigation reports, 2025–2026 (recommended)

- **What:** NTSB Marine Investigation Reports (`MIR-25-01` … `MIR-25-47`, `MIR-26-01` … `MIR-26-30`). These are 77 PDFs found at `https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR25NN.pdf`. Each report covers one casualty: a grounding, fire, collision, flooding and so on.
- **License evidence:** NTSB website policy (https://www.ntsb.gov/about/Policies/Pages/Policies.aspx), "Text Content":
  > "Text appearing on NTSB Web pages, in reports, recommendation, and public dockets, unless otherwise noted, was prepared by employees of the United States Government as part of their official duties and, therefore, is not subject to copyright." … "By publishing its reports and recommendations and opening public dockets, the NTSB has placed the contents prepared by its staff in the public domain."

  Caveats from the same policy: the NTSB logo and seal are excluded, and third-party copyrighted material is identified as such. We would use report text only, with a "Courtesy: National Transportation Safety Board" credit.
- **Memorisation:** low. Reports were published from 2025 on (e.g. MIR-25-30 on 2025-07-22). The incidents themselves may have had news coverage, but the report details (casualty-summary tables, probable cause, timings) are new.
- **Depth:** a sampled report (MIR-25-30) has 11 pages and ~2,300 words. Every report has the same structure: casualty summary table, narrative, analysis, probable cause, lessons learned. That makes for natural comparison and multi-document questions ("Which 2025 groundings involved…", "Compare the probable causes of…") and plausible unanswerable ones.
- **Practical notes:** the PDFs are encrypted with an owner password, but text extraction works (checked with pypdf). Parsing tables and figures through AnythingLLM's collector is realistic noise, but it is noise. Smaller documents also mean more chunk boundaries to test.

### B. Python Enhancement Proposals created in 2025–2026

- **What:** the ~25 PEPs with status Final or Accepted that were created in 2025 or later. Examples: PEP 784 (Zstandard in the stdlib), PEP 791 (`math.integer`), PEP 798 (unpacking in comprehensions), PEP 810 (explicit lazy imports). The `.rst` sources are in https://github.com/python/peps, about 90k words in total.
- **License evidence:** each PEP has its own footer, for example PEP 810:
  > "This document is placed in the public domain or under the CC0-1.0-Universal license, whichever is more permissive."

  I checked 25 recent PEPs: 24 carry this footer and **PEP 772 has no license statement**, so it would be excluded. The repository itself declares no license, so the license must be checked per document.
- **Memorisation:** low to moderate. The PEPs postdate the model, but the model knows Python well, and some proposals revive older ideas (lazy imports were already proposed in PEP 690, rejected in 2022). That conflict is useful for the knowledge-edit mutation, but it raises the closed-book rate.
- **Depth:** long documents (1k–13k words) with heavy cross-referencing, e.g. the C-API and ABI PEPs 782/793/803. This suits procedural and multi-document questions.
- **Practical notes:** clean plain-text sources, so the least parsing noise. The questions are developer-oriented, which suits an engineering audience.

### C. NIST SP 800-63-4, Digital Identity Guidelines (final, July 2025)

- **What:** four volumes (63-4 base, 63A-4 identity proofing, 63B-4 authentication, 63C-4 federation). The HTML/Markdown source is at https://github.com/usnistgov/800-63-4.
- **License evidence:** NIST statement on technical series publications (https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications):
  > "Works authored by NIST employees are not subject to Copyright protection within the United States"

  Outside the US, NIST grants "the non-exclusive, perpetual, paid-up, royalty-free, worldwide right to reprint works". The requested credit line is "Republished courtesy of the National Institute of Standards and Technology." The GitHub repository has no license file.
- **Memorisation:** **moderate to high.** Revision 3 (2017) is widely known, and public drafts of Revision 4 appeared before 2025. The model is likely to answer some questions from memory, and to answer some with outdated Revision 3 requirements.
- **Depth:** very deep normative text ("SHALL/SHOULD" requirements) spread across volumes, with strong multi-document and procedural potential.

## Recommendation

**A (NTSB 2025–2026)** fits the requirements best. It has the lowest memorisation risk, many uniform documents (good for stratified splits and comparison questions), clear public-domain status, and realistic PDF ingestion. **B** is a close second if you prefer clean text and a developer-facing domain. **C** is the weakest on memorisation.

## Decision

**A: NTSB marine investigation reports, 2025–2026**, chosen by the repository owner on 2026-10-21.

Text is used as published, with the credit "Courtesy: National Transportation Safety Board". The NTSB logo and seal, and any material the reports mark as third-party copyrighted, are not covered by the public-domain statement and are not relied on.

## Consequences

- **Storage:** the PDFs (77 files, ~110 MB) are not committed. `corpus/sources.yaml` pins each report's URL and SHA-256 (taken 2026-10-21), and `rag-eval fetch-corpus` downloads and verifies them. A changed file is rejected rather than silently changing the corpus, which matters because NTSB reissues corrected reports (e.g. MIR-25-23, MIR-25-34 and MIR-25-43 were reissued in August 2026). The README links every report.
- **Uneven document sizes:** most reports are 7–29 pages, but four are long: MIR-25-40 (Dali / Francis Scott Key Bridge, 259 pages), MIR-25-16 (95), MIR-26-24 (92) and MIR-25-36 (Titan submersible, 87). Questions should not be drawn mostly from these.
- **Memorisation hot spots:** the Dali and Titan casualties had worldwide news coverage before the reports came out. Questions about them are the most likely to be answered from memory. The closed-book baseline will show whether they need to be down-weighted or excluded (with `corpus.exclude`).

