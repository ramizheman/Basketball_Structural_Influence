# Structural Influence: Measuring the Hidden Wiring of an NBA Offense

**Independent verification of reported statistics uses frozen, analysis-ready tables in `output/` (Layer 1). Rebuilding the possession graph from raw NBA Stats API pulls needs the author's Neo4j warehouse and is not required for verification.**

Code and analysis-ready data for the manuscript *Structural Influence: Measuring the Hidden Wiring of an NBA Offense* (Rami Zheman, 2026).

## Data

- **Source:** public NBA Stats API (`stats.nba.com`) via open-source [`nba_api`](https://github.com/swar/nba_api).
- **Seasons (confirmatory):** 2022–23, 2023–24, 2024–25 regular season.
- **Exploratory addendum:** 2025–26 shaper roster in Appendix G (`shaper_appendix_2526.csv`).
- **No proprietary data:** no Synergy / Second Spectrum / tracking feeds.
- **Warehouse:** the author's possession graph (Neo4j) was used to create `output/`. The public path starts from that dump.

## Reproduce headline results (Layer 1, no Neo4j)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Verify locked numeric claims against output/ caches (must exit 0)
python _paper_numbers_audit.py --check

# Optional: regenerate paper figures into figures/
python make_figures.py
```

Manuscript: `paper.tex`.  
Traceability sheet (written by the audit): `PAPER_NUMBERS_SHEET.md`.  
Possession vs initiation edge sources: `DATA_SOURCE_AUDIT.md`.

## Claim → paper section → code

Section numbers match `paper.tex` (article class). Figures are regenerated with `make_figures.py`.

| Paper section | Claim | Script | Artifact |
|---------------|-------|--------|----------|
| §3 Data | Corpus size (~6.5×10⁵ possessions; ON_COURT-5) | `role_composition_playmix.py` | `role_composition_possessions.parquet` |
| §4.2 Association matrix | CUT–SPOT_UP min-overlap worked example (MIN 2023–24) | `section4_worked_example.py` | `section4_cut_spotup_example.csv` |
| §4.3 / Fig. 3 | Leave-one-out displacement (Gobert vs Towns) | `make_figures.py` | `fig_loo_displacement.png` |
| §4.5 Role categorizations | Usage × residual cut that defines Shapers | `structural_role_taxonomy.py` | `structural_role_*.csv` |
| §6.1 Does the measure reveal something new? | Taxonomy table (n=924) | `structural_role_taxonomy.py` | `structural_role_*.csv` |
| §6.1 / Fig. 4 | Named exemplars / Organizer–Terminal map | `make_figures.py` | `fig_taxonomy_map.png` |
| §6.2 Is it just another value metric? | Orthogonality vs net rating / box stats | `fragility_orthogonality.py` | `fragility_orthogonality.csv` |
| §6.3 Can we actually see what these shapers do? | Play-mix on/off deltas | `role_composition_playmix.py` | `role_composition_playmix.csv` |
| §6.3 | Route entropy (rebalance, not broaden) | `role_route_graph.py` | `role_route_graph.csv` |
| §6.4 Where do we stop? | Coverage null; organizer accumulation +2.35 | `coverage_full_study.py` | `coverage_full_study.csv` |
| §6.4 | HasShaper null | `hasshaper_full_study.py` | `hasshaper_full_study.csv` |
| §6.4.1 / Appendix F | Exploratory shaper-tier PPP split | `shaper_tier_full_study.py` | `shaper_tier_full_study.csv` |
| §6.4.2 Identity still changes? | On/off \(D_A\) (disclosed post-hoc) | `_identity_resilience_test.py` | `identity_resilience_*.csv` |
| §6.5 Partly player-specific? | Portability (n=294 movers) | `portability_*.py` | `portability_*.csv` / `.txt` |
| §6.5 / Fig. 5–6 | Shaper \(\Delta\) distribution and concordance maps | `make_figures.py` | `fig_shaper_delta_dist.png`, `fig5b_signatures.png` |
| §6.6 Can transfer be predicted? | Composite travel score (AUC 0.61) | `travel_fit_screen.py` | `travel_fit_screen.csv` |
| §6.6 / Appendix C | Cut vs drive screen (exploratory) | `cut_drive_travel_screen.py` | `cut_drive_travel_screen.csv` |
| §6.6 / Appendix C | Destination-side FDR (63 features) | (frozen table) | `destination_battery_shapers.csv` |
| Appendix G | 2025–26 Shaper roster (exploratory) | (frozen table) | `shaper_appendix_2526.csv` |
| Appendix H | Play-type inferrer from public PBP | (manuscript) | — |
| Appendix I | Play-type pair glossary | (manuscript) | — |

Pre-registration protocols for the paper's pre-registered analyses: `PORTABILITY_PREREGISTRATION.md`, `TOPOLOGY_COVERAGE_PREREGISTRATION.md`.

## License / use

Research code accompanying the paper. NBA data remain subject to NBA Stats terms of use.
