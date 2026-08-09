# Data-source audit (confirmatory scripts)

_Generated 2026-08-09. Report only — no scripts were modified._

Context: `role_composition_possessions.parquet` (and the `_by_id` twin) require
exactly-5 offensive `ON_COURT` players and **undercount** initiations relative to
`full_edges.parquet` (L2 `INITIATED` edges). Example: Gobert MIN 2023–24 CUT is
245 in the possession cache vs **277** in `full_edges` (paper ≈62% cut share uses
the edge count: 277/445).

## Legend

| Flag | Meaning |
|------|---------|
| **EDGE** | Reads `full_edges.parquet` / `pull_edges()` (INITIATED). Correct for B / association / influence. |
| **ON_COURT** | Reads `role_composition_possessions*.parquet` (exactly-5 ON_COURT). Known undercount vs edges for raw initiation totals. |
| **DERIVED** | Reads frozen CSVs built upstream (no direct possession parquet). Upstream source noted. |
| **LOCKED** | Numbers appear in `PAPER_NUMBERS_SHEET.md`. |

---

## Confirmatory / locked-claim scripts

| Script | Locked in sheet? | Primary data source | Flag | Notes |
|--------|------------------|---------------------|------|-------|
| `wiring_full_study.py` | (feeds loads) | `full_edges.parquet` via `pull_edges()` | **EDGE** | Builds L2 INITIATED cache from Neo4j if missing. |
| `fragility_full_study.py` | via loads / taxonomy | `pull_edges()` → `full_edges.parquet` | **EDGE** | Writes `fragility_full_loads.csv`. |
| `structural_role_taxonomy.py` | **LOCKED** §6.1 | `fragility_full_loads.csv`, `player_traditional_stats.csv` | **DERIVED** ← EDGE | No possession parquet. |
| `fragility_orthogonality.py` | **LOCKED** §6.2 | `fragility_full_loads.csv` + box stats | **DERIVED** ← EDGE | No possession parquet. |
| `role_composition_playmix.py` | **LOCKED** §Data, §6.3 | `role_composition_possessions.parquet` via `pull_possessions()` | **ON_COURT** | Cypher: `ON_COURT` + `size(oncourt)=5`. Also writes that cache. |
| `role_route_graph.py` | **LOCKED** §6.3 entropy | `pull_possessions()` → same parquet | **ON_COURT** | Needs lineup slots for role composition. |
| `coverage_full_study.py` | **LOCKED** §6.4 | `pull_possessions()` → same parquet | **ON_COURT** | COVERED / PPP; needs on-court roles. |
| `coverage_gate.py` | gate (not sheet) | `pull_possessions()` | **ON_COURT** | Same corpus as coverage study. |
| `hasshaper_full_study.py` | **LOCKED** §6.4 HasShaper | `pull_possessions()` → same parquet | **ON_COURT** | Same machinery as coverage. |
| `shaper_tier_full_study.py` | **LOCKED** (exploratory tiers) | `pull_possessions()` → same parquet | **ON_COURT** | |
| `_identity_resilience_test.py` | **LOCKED** §6.4.2 | `pull_possessions_by_id()` → `role_composition_possessions_by_id.parquet` | **ON_COURT** | Same exactly-5 ON_COURT filter; builds on/off **B** from that filtered corpus (not `full_edges`). |
| `portability_full_study.py` | **LOCKED** §6.5 | `portability_signatures.parquet` via `build_signature_cache()` | **DERIVED** ← EDGE | Signatures built from `pull_edges()` / `full_edges.parquet`. |
| `portability_gate.py` | **LOCKED** vacuity | signatures cache ← edges | **DERIVED** ← EDGE | |
| `portability_common.py` | (library) | `pull_edges()` → signatures | **EDGE** | |
| `travel_fit_screen.py` | **LOCKED** §6.6 | `portability_players_enriched.csv`, `portability_signatures.parquet` | **DERIVED** ← EDGE | |
| `cut_drive_travel_screen.py` | **LOCKED** §6.6 | `travel_fit_screen.csv`, `portability_signatures.parquet` | **DERIVED** ← EDGE | |
| `section4_worked_example.py` | **LOCKED** §4.2 (new) | `full_edges.parquet` | **EDGE** | Worked-example CUT/SPOT_UP counts. |

---

## Explicit flags: locked scripts on the undercounting ON_COURT cache

These confirmatory scripts (locked numbers in `PAPER_NUMBERS_SHEET.md`) **do** read
`role_composition_possessions.parquet` or `role_composition_possessions_by_id.parquet`:

1. **`role_composition_playmix.py`** — source: `role_composition_possessions.parquet` — **undercounts vs INITIATED**
2. **`role_route_graph.py`** — source: same via `pull_possessions()` — **undercounts vs INITIATED**
3. **`coverage_full_study.py`** — source: same — **undercounts vs INITIATED**
4. **`hasshaper_full_study.py`** — source: same — **undercounts vs INITIATED**
5. **`shaper_tier_full_study.py`** — source: same — **undercounts vs INITIATED**
6. **`_identity_resilience_test.py`** — source: `role_composition_possessions_by_id.parquet` — **undercounts vs INITIATED**

**Not modified.** Human decision: whether any locked §6.3 / §6.4 / §6.4.2 numbers need
re-verification against a non-ON_COURT corpus.

### Design note (for the human reviewer)

ON_COURT-5 is **required** for analyses that need the five on-floor players
(play-mix with/without a class, COVERED, HasShaper, route entropy). Absolute
initiation undercount vs `full_edges` does not by itself prove those Δshare / PPP
coefficients are wrong — they are estimated inside the ON_COURT-5 sample by design.
The sharper concern is **`_identity_resilience_test.py`**, which constructs
association matrices \(A\) / \(D_A\) from the ON_COURT-filtered possession set, whereas
§4 association / structural influence use INITIATED edges (`full_edges`).

Corpus size claim (§Data ≈6.5e5) is explicitly the ON_COURT-5 count (651784) and
matches that cache, not the full INITIATED edge volume.

---

## Scripts that correctly use EDGE for B / influence / portability

- `wiring_full_study.py` / `fragility_full_study.py` → loads
- `structural_role_taxonomy.py` / `fragility_orthogonality.py` (derived)
- `portability_*` / `travel_fit_screen.py` / `cut_drive_travel_screen.py` (via signatures)
- `section4_worked_example.py` (new)
