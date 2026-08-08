# Structural Influence: Measuring the Hidden Wiring of an NBA Offense

**Independent verification of reported statistics uses frozen, analysis-ready tables in `output/` (Layer 1). Rebuilding the possession graph from raw NBA Stats API pulls needs the author's Neo4j warehouse and is not required for verification.**

Code and analysis-ready data for the manuscript *Structural Influence: Measuring the Hidden Wiring of an NBA Offense* (Rami Zheman, 2026).

## Data

- **Source:** public NBA Stats API (`stats.nba.com`) via open-source [`nba_api`](https://github.com/swar/nba_api).
- **Seasons (confirmatory):** 2022–23, 2023–24, 2024–25 regular season.
- **Exploratory addendum:** 2025–26 shaper roster in Appendix G (`connector_appendix_2526.csv`).
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

## Claim → code map

| Paper claim | Script | Artifact |
|-------------|--------|----------|
| Taxonomy (n=924) | `structural_role_taxonomy.py` | `structural_role_*.csv` |
| Orthogonality | `fragility_orthogonality.py` | `fragility_orthogonality.csv` |
| Play-mix / route entropy | `role_composition_playmix.py`, `role_route_graph.py` | `role_composition_playmix.csv`, `role_route_graph.csv` |
| Covered / +2.35 organizers | `coverage_full_study.py` | `coverage_full_study.csv` |
| HasShaper null | `hasconn_full_study.py` | `hasconn_full_study.csv` |
| Shaper tiers (exploratory) | `connector_tier_full_study.py` | `connector_tier_full_study.csv` |
| Identity \(D_A\) | `_identity_resilience_test.py` | `identity_resilience_*.csv` |
| Portability (n=294) | `portability_*.py` | `portability_*.csv` / `.txt` |
| Travel score (AUC 0.61) | `travel_fit_screen.py` | `travel_fit_screen.csv` |
| Cut vs drive (exploratory) | `cut_drive_travel_screen.py` | `cut_drive_travel_screen.csv` |
| Destination-side FDR (63 features) | (frozen table) | `destination_battery_connectors.csv` |
| Appendix G 2025–26 shapers | (frozen table) | `connector_appendix_2526.csv` |

Pre-registration protocols for the paper's pre-registered analyses: `PORTABILITY_PREREGISTRATION.md`, `TOPOLOGY_COVERAGE_PREREGISTRATION.md`.

## License / use

Research code accompanying the paper. NBA data remain subject to NBA Stats terms of use.
