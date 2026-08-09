"""
Section 4.2 worked example — MIN 2023–24 CUT / SPOT_UP initiation counts
=======================================================================

Reproduces the verified player-level initiation counts that back the
min-overlap co-initiation worked example in ``paper.tex`` Section 4.2
(association matrix / Eq. overlap).

Source of truth: ``output/full_edges.parquet`` (L2 ``INITIATED`` edges from
``wiring_full_study.pull_edges``). This is the same B-matrix construction used
for structural influence / association matrices.

Do NOT use ``role_composition_possessions.parquet`` here: that cache requires
exactly-5 ON_COURT offensive players and undercounts initiations relative to
the L2 edge graph (e.g. Gobert CUT 245 vs 277).

Rotation pool gate (matches the paper elsewhere): mpg >= 15, gp >= 30.

Outputs:
  output/section4_cut_spotup_example.csv

Also prints Gobert's team cut share (paper ≈62%; exact 277/445 = 62.25%).

Run: python section4_worked_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
EDGES = OUT_DIR / "full_edges.parquet"
STATS = OUT_DIR / "player_traditional_stats.csv"
DEST = OUT_DIR / "section4_cut_spotup_example.csv"

PLAY8 = [
    "TRANSITION",
    "PNR",
    "DRIVE",
    "POST_UP",
    "SPOT_UP",
    "CUT",
    "PUTBACK",
    "PULL_UP",
]
TEAM, YY = "MIN", 23
MIN_MPG, MIN_GP = 15.0, 30


def main() -> None:
    if not EDGES.exists():
        raise SystemExit(
            f"Missing {EDGES.name}. Build via wiring_full_study.pull_edges() / Neo4j."
        )
    edges = pd.read_parquet(EDGES)
    stats = pd.read_csv(STATS)

    rot = stats[
        (stats["tri"] == TEAM)
        & (stats["yy"] == YY)
        & (stats["mpg"] >= MIN_MPG)
        & (stats["gp"] >= MIN_GP)
    ][["abbrev", "full_name", "gp", "mpg"]].copy()

    rs = edges[
        (edges["team"] == TEAM)
        & (edges["season_yy"] == YY)
        & (edges["is_playoff"] == 0)
        & (edges["ptype"].isin(PLAY8))
    ].copy()

    piv = (
        rs.groupby(["player", "ptype"])["n"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=PLAY8, fill_value=0)
    )
    piv["total_initiations"] = piv.sum(axis=1)
    out = piv.loc[piv.index.isin(set(rot["abbrev"])), ["CUT", "SPOT_UP", "total_initiations"]].copy()
    out = out.reset_index().rename(columns={"player": "player"})
    out = out.merge(rot, left_on="player", right_on="abbrev", how="left")
    out["min_CUT_SPOT_UP"] = np.minimum(out["CUT"], out["SPOT_UP"]).astype(int)
    out["CUT"] = out["CUT"].astype(int)
    out["SPOT_UP"] = out["SPOT_UP"].astype(int)
    out["total_initiations"] = out["total_initiations"].astype(int)
    out = out.sort_values("CUT", ascending=False)

    cols = [
        "player",
        "full_name",
        "gp",
        "mpg",
        "CUT",
        "SPOT_UP",
        "min_CUT_SPOT_UP",
        "total_initiations",
    ]
    out[cols].to_csv(DEST, index=False)

    team_cut = int(rs.loc[rs["ptype"] == "CUT", "n"].sum())
    gob_cut = int(out.loc[out["player"] == "R. Gobert", "CUT"].iloc[0])
    share = gob_cut / team_cut if team_cut else float("nan")

    print(out[cols].to_string(index=False))
    print(f"\nWrote {DEST}")
    print(
        f"Gobert CUT share of MIN 2023-24 RS team CUT: "
        f"{gob_cut}/{team_cut} = {100 * share:.2f}%"
    )


if __name__ == "__main__":
    main()
