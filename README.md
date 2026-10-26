# rag-evaluation-reference

Reference implementation of RAG evaluation: error analysis, validated binary LLM judges, mutation testing and statistically sound results. System under test: AnythingLLM, with DeepEval.

> **Status:** work in progress. Provisioning, the runner, the dataset and the closed-book baseline are in place; the evaluators and results are not yet.

## Quick start

Requirements: Docker, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) running natively on the host (on Apple Silicon this gives the models GPU acceleration; AnythingLLM in Docker reaches it via `host.docker.internal`).

```bash
# 1. Models (digests are pinned in configs/baseline.yaml and verified on every run)
ollama pull qwen3:8b
ollama create qwen3-nothink:8b -f models/qwen3-nothink.Modelfile   # thinking off (ADR 0005)
ollama pull nomic-embed-text

# 2. AnythingLLM (pinned image, localhost only)
docker compose up -d

# 3. API key and environment (see .env.example)
cp .env.example .env   # then set ANYTHINGLLM_API_KEY as described in the file
set -a; source .env; set +a

# 4. Fetch and verify the corpus (~110 MB from ntsb.gov)
uv sync
uv run rag-eval fetch-corpus

# 5. Provision and run (smoke test on two synthetic documents)
uv run rag-eval provision -c tests/smoke/smoke.yaml
uv run rag-eval run -c tests/smoke/smoke.yaml -d tests/smoke/cases.jsonl
```

Each run writes `results/<run-id>/traces.jsonl` (one record per case) and `manifest.json` (git commit, config hash, pinned image and model digests, dataset and corpus hashes).

## Corpus

The system is evaluated on the **77 NTSB Marine Investigation Reports published in 2025 and 2026**: public-domain US government reports, each on one maritime casualty, chosen because they postdate the model under test (see [ADR 0003](docs/decisions/0003-corpus-choice.md)). Courtesy: National Transportation Safety Board; see [`corpus/ATTRIBUTION.md`](corpus/ATTRIBUTION.md).

The PDFs are not committed. `uv run rag-eval fetch-corpus` downloads them from ntsb.gov into `corpus/documents/` and verifies each against the SHA-256 pinned in [`corpus/sources.yaml`](corpus/sources.yaml).

<details>
<summary>All 77 reports</summary>

| Report | Title |
|---|---|
| [MIR-25-01](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2501.pdf) | Contact of Kitty Tow with Algiers Lock Gate |
| [MIR-25-02](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2502.pdf) | Contact of Crane on Deck Barge BMI 209 Towed by Nickelena with Mackinac Bridge |
| [MIR-25-03](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2503.pdf) | Grounding of Cargo Vessel Bonnie G |
| [MIR-25-04](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2504.pdf) | Engine Room Fire aboard Fishing Vessel Three Girls |
| [MIR-25-05](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2505.pdf) | Engine Room Fire aboard Cargo Vessel Stride |
| [MIR-25-06](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2506.pdf) | Container Loss aboard Containership President Eisenhower |
| [MIR-25-07](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2507.pdf) | Contact of Small Passenger Vessel Columbia Gorge with Pedestrian Bridge and Tribal Fishing Platforms |
| [MIR-25-08](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2508.pdf) | Collapse of Chalmette Refinery Dock Section During Docking of Ovide J Tow |
| [MIR-25-09](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2509.pdf) | Contact of Crane on board Royal Engineer Tow with Terminal Cargo Crane |
| [MIR-25-10](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2510.pdf) | Safeguarding Bridges from Vessel Strikes: Need for Vulnerability Assessment and Risk Reduction Strategies |
| [MIR-25-11](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2511.pdf) | Contact of Tank Vessel Hafnia Amessi with Pier |
| [MIR-25-12](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2512.pdf) | Collision between Fishing Vessel Papa Rod and Anchored Bulk Carrier Appaloosa |
| [MIR-25-13](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2513.pdf) | Diesel Generator Engine Failure aboard Passenger Vessel Ocean Navigator |
| [MIR-25-14](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2514.pdf) | Contact of Amber Brittany Tow with Vane Dike |
| [MIR-25-15](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2515.pdf) | Engine Room Fire aboard Towing Vessel Johnny M |
| [MIR-25-16](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2516.pdf) | Fire aboard Roll-on/Roll-off Container Vessel Grande Costa D’Avorio |
| [MIR-25-17](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2517.pdf) | Contact of Towing Vessel Cocodrie with Valero St. Charles Refinery Walkway |
| [MIR-25-18](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2518.pdf) | Contact of Bulk Carrier American Mariner with Munuscong Channel Junction Light |
| [MIR-25-19](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2519.pdf) | Grounding of Towing Vessel City of Louisville |
| [MIR-25-20](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2520.pdf) | Fire aboard Fishing Vessel Jordyn Rose |
| [MIR-25-21](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2521.pdf) | Capsizing and Sinking of Towing Vessel Baylor J. Tregre |
| [MIR-25-22](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2522.pdf) | Sinking of the Industrial Vessel Cape Douglas |
| [MIR-25-23](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2523.pdf) | Grounding of Towing Vessel Tom Frazier |
| [MIR-25-24](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2524.pdf) | Fire aboard Fishing Vessel Tarka II |
| [MIR-25-25](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2525.pdf) | Contact of Duke Tow with Power Transmission Poles and Pelican Island Bridge |
| [MIR-25-26](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2526.pdf) | Collision of William B Klunk Tow with Moored Barges |
| [MIR-25-27](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2527.pdf) | Fire aboard Towing Vessel Lucinda Smith |
| [MIR-25-28](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2528.pdf) | Contact of Joe B. Wyatt Tow with Fort Madison Bridge Protection Cell and Fendering System |
| [MIR-25-29](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2529.pdf) | Listing and Partial Sinking of Yacht Lovebug |
| [MIR-25-30](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2530.pdf) | Grounding of Recreational Vessel 6 Saturdays |
| [MIR-25-31](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2531.pdf) | Fire aboard Yacht Flagship |
| [MIR-25-32](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2532.pdf) | Grounding of Barge Cordova Provider |
| [MIR-25-33](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2533.pdf) | Collision of Multi-Purpose Carrier BBC Africa with Bulk Carrier Common Faith |
| [MIR-25-34](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2534.pdf) | Contact of Ferry Ray Stoker, Jr. with Floating Dock |
| [MIR-25-35](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2535.pdf) | Contact of Tanker Cosflourish Lake with South Texas Gateway Terminal Dock |
| [MIR-25-36](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2536.pdf) | Hull Failure and Implosion of Submersible Titan, North Atlantic Ocean, 370 nautical miles southeast of St. John’s, Newfoundland and Labrador, Canada, June 18, 2023 |
| [MIR-25-37](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2537.pdf) | Grounding of the Towing Vessel John M Donnelly |
| [MIR-25-38](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2538.pdf) | Fire aboard Bulk Carrier Cuyahoga |
| [MIR-25-39](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2539.pdf) | Engine Room Fire aboard Dredging Vessel Stuyvesant |
| [MIR-25-40](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2540.pdf) | Contact of Containership Dali with Francis Scott Key Bridge and Subsequent Bridge Collapse |
| [MIR-25-41](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2541.pdf) | Lithium-ion Battery Fires aboard Cargo Vessel Genius Star XI |
| [MIR-25-42](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2542.pdf) | Collision between Offshore Supply Vessel Jack Edwards and Cargo Vessel Julie C |
| [MIR-25-43](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2543.pdf) | Collision Between Towing Vessel William E Strait and Moored Barges at Shell Norco Refinery Dock |
| [MIR-25-44](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2544.pdf) | Flooding and Sinking of Towing Vessel Cajole |
| [MIR-25-45](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2545.pdf) | Contact of Bulk Carrier Algoma Discovery with BNSF Allouez Taconite Facility |
| [MIR-25-46](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2546.pdf) | Sinking of Barge PTC 706 and Subsequent Breakup of Chad Pregracke Tow |
| [MIR-25-47](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2547.pdf) | Contact of Towing Vessel Schweiger with US Coast Guard Station Cape Disappointment Docks |
| [MIR-26-01](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2601.pdf) | Engine Room Fire on Towing Vessel Thor |
| [MIR-26-02](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2602.pdf) | Diesel Generator Engine Failure aboard Self unloading Barge Pathfinder |
| [MIR-26-03](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2603.pdf) | Flooding of Bulk Carrier Cuyahoga |
| [MIR-26-04](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2604.pdf) | Fire aboard Fishing Vessel Spicy Lady |
| [MIR-26-05](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2605.pdf) | Contact of Tanker Platanos with Martinez Refining Company Wharf Pier |
| [MIR-26-06](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2606.pdf) | Grounding of Dry Bulk Carrier Algoma Verity |
| [MIR-26-07](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2607.pdf) | Grounding of Fishing Vessel Eileen Rita |
| [MIR-26-08](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2608.pdf) | Engine Room Fire aboard Cargo Vessel Lem Verbena |
| [MIR-26-09](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2609.pdf) | Engine Room Flooding aboard Sailing School Vessel Oliver Hazard Perry |
| [MIR-26-10](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2610.pdf) | Breakaway of Barges MPC 907 and MPC 644 and Subsequent Contact with Mooring Cells |
| [MIR-26-11](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2611.pdf) | Crane Wire Failure on Offshore Construction Vessel Island Venture and Subsequent Damage to Offshore Supply Vessel C-Enforcer |
| [MIR-26-12](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2612.pdf) | Fire aboard Pipelayer Deep Energy |
| [MIR-26-13](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2613.pdf) | Collision between Towing Vessel Patrick J Studdert and Bulk Carrier Clara B and Subsequent Sinking of Patrick J Studdert |
| [MIR-26-14](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2614.pdf) | Contact of LCPL Phillip C George Tow with Terminal Dolphin, Power Transmission Pole, and Pelican Island Bridge |
| [MIR-26-15](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2615.pdf) | Contact of Mackenzie Rose Tow with Norfolk and Portsmouth Belt Line Railroad Bridge |
| [MIR-26-16](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2616.pdf) | Flooding and Sinking of Fishing Vessel North American |
| [MIR-26-17](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2617.pdf) | Contact of Rickey Hughes Tow with the Vicksburg Highway 80 Bridge |
| [MIR-26-18](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2618.pdf) | Contact of Josset Tow with the Bayou Grosse Tête Bridge |
| [MIR-26-19](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2619.pdf) | Collapse of Emissions Control Barge STAX 1 Capture and Control Articulated Arm |
| [MIR-26-20](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2620.pdf) | Contact of Myra Eckstein Tow with Vicksburg Highway 80 Bridge |
| [MIR-26-21](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2621.pdf) | Fire aboard Towing Vessel William D |
| [MIR-26-22](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2622.pdf) | Explosion aboard Sludge Vessel Hunts Point |
| [MIR-26-23](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2623.pdf) | Fire aboard Fishing Vessel Sub Zero |
| [MIR-26-24](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2624.pdf) | Collision Between Bulk Carrier Yangze 7 and Towing Vessel Miss Peggy |
| [MIR-26-25](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2625.pdf) | Fire aboard Fishing Vessel GITN’R DUN |
| [MIR-26-26](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2626.pdf) | Contact of Bulk Carrier Belle Mer Anchor with Subsea Cable |
| [MIR-26-27](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2627.pdf) | Contact of Offshore Supply Vessel Dry Tortugas with Offshore Gas and Condensate Production Platform VK-817A |
| [MIR-26-28](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2628.pdf) | Flooding and Sinking of Charter Fishing Vessel Jig Strike |
| [MIR-26-29](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2629.pdf) | Engine Room Fire aboard Chemical Tanker Scot Stuttgart |
| [MIR-26-30](https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2630.pdf) | Grounding of Fishing Vessel Mystery |

</details>

## Dataset

110 questions about the reports (44 dev, 66 test), each with verbatim evidence quotes, built with `rag-eval dataset ...` and explained in [ADR 0004](docs/decisions/0004-dataset-size-and-generation.md). Known weak items are listed in [`data/candidates/review-notes.md`](data/candidates/review-notes.md).

Self-contained pages show the human-in-the-loop steps; download and open them in a browser:

- [`data/review/review.html`](data/review/review.html): reviewing generated candidates (approve, edit or reject, with each quote shown in its report context).
- [`data/review/traces.html`](data/review/traces.html): open coding for error analysis, with each trace's expected evidence highlighted in the retrieved passages (traces shuffled, config hidden).
- [`data/review/closed-book.html`](data/review/closed-book.html): marking the model's answers with no documents. The marks made with it are in [`data/labels/closed-book/`](data/labels/closed-book/) and the result in [`docs/results/closed-book-baseline.md`](docs/results/closed-book-baseline.md): 3 of 90 questions answered correctly from memory, all of them guessable.

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```

Design notes: [`docs/spike-findings.md`](docs/spike-findings.md) and [`docs/decisions/`](docs/decisions/).
