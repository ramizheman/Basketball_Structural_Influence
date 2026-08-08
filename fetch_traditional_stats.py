"""
Fetch REAL traditional + advanced + tracking stats per player-season from the
NBA Stats API, for the orthogonality argument (structural influence vs conventional
importance). Official NBA data, matched by team-season to our graph names.

Pulled per season (each endpoint returns all players in one call):
  Base      : PTS, AST, MIN, GP
  Advanced  : USG_PCT, NET_RATING (on-court net rating)
  Tracking  : TOUCHES, FRONT_CT_TOUCHES (Possessions measure)

Writes: output/player_traditional_stats.csv (tri, yy, abbrev, ...stats)

Run: python fetch_traditional_stats.py
"""
from __future__ import annotations

import sys
import time

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashptstats

from wiring_full_study import OUT_DIR
from fragility_tables import season_label

OUT_CSV = OUT_DIR / "player_traditional_stats.csv"
SEASONS_YY = [22, 23, 24, 25]


def to_abbrev(full_name: str) -> str:
    name = full_name.replace("\u2019", "'").strip()
    parts = name.split()
    if len(parts) < 2:
        return name
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def main():
    frames = []
    for yy in SEASONS_YY:
        season = season_label(yy)
        print(f"--- {season} ---", flush=True)
        base = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, season_type_all_star="Regular Season",
            measure_type_detailed_defense="Base", per_mode_detailed="PerGame",
            timeout=45).get_data_frames()[0]
        time.sleep(0.6)
        adv = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame",
            timeout=45).get_data_frames()[0]
        time.sleep(0.6)
        pt = leaguedashptstats.LeagueDashPtStats(
            season=season, season_type_all_star="Regular Season",
            pt_measure_type="Possessions", player_or_team="Player",
            per_mode_simple="PerGame", timeout=45).get_data_frames()[0]
        time.sleep(0.6)

        b = base[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "PTS", "AST"]]
        a = adv[["PLAYER_ID", "USG_PCT", "NET_RATING", "AST_PCT"]]
        t = pt[["PLAYER_ID", "TOUCHES", "FRONT_CT_TOUCHES"]]
        m = b.merge(a, on="PLAYER_ID", how="left").merge(t, on="PLAYER_ID", how="left")
        m["yy"] = yy
        m["season"] = season
        m["tri"] = m["TEAM_ABBREVIATION"]
        m["abbrev"] = m["PLAYER_NAME"].map(to_abbrev)
        frames.append(m)
        print(f"  {len(m)} players", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"PLAYER_NAME": "full_name", "GP": "gp", "MIN": "mpg",
                              "PTS": "ppg", "AST": "apg", "USG_PCT": "usg_pct",
                              "NET_RATING": "net_rating", "AST_PCT": "ast_pct",
                              "TOUCHES": "touches", "FRONT_CT_TOUCHES": "front_ct_touches"})
    cols = ["tri", "yy", "season", "abbrev", "full_name", "gp", "mpg", "ppg", "apg",
            "usg_pct", "ast_pct", "net_rating", "touches", "front_ct_touches"]
    out[cols].to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out)} rows)")


if __name__ == "__main__":
    main()
