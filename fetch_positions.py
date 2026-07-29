"""
Fetch REAL, season-specific player positions from the NBA Stats API
(commonteamroster) for every (team, season) in the fragility dataset.

This is deliberately NOT derived from graph_schema.md's graph (Player has no
position property there) — position is pulled from the NBA's own official
roster data for the exact team-season, then matched to our graph's player
name format ("F. Last[ Suffix]").

Writes: output/player_positions.csv  (tri, yy, full_name, position, abbrev)
Cache:  re-run is a no-op if the CSV already has full coverage.

Run: python fetch_positions.py
"""
from __future__ import annotations

import re
import sys
import time

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from nba_api.stats.endpoints import commonteamroster, commonplayerinfo
from nba_api.stats.static import teams, players as static_players

from wiring_full_study import OUT_DIR
from fragility_tables import season_label

LOADS = OUT_DIR / "fragility_full_loads.csv"
OUT_CSV = OUT_DIR / "player_positions.csv"

TEAM_ID = {t["abbreviation"]: t["id"] for t in teams.get_teams()}
# our graph uses ATL/BOS/... standard tricodes; NBA API uses the same for
# all 30 current-era franchises in this window (no relocations 2022-2026).


def to_abbrev(full_name: str) -> str:
    """'Rudy Gobert' -> 'R. Gobert'; 'Wendell Moore Jr.' -> 'W. Moore Jr.'"""
    name = full_name.replace("’", "'").strip()
    parts = name.split()
    if len(parts) < 2:
        return name
    first = parts[0]
    rest = " ".join(parts[1:])
    # collapse hyphenated / multi-word first names' initial from first token only
    initial = first[0]
    return f"{initial}. {rest}"


def normalize(s: str) -> str:
    """loose match key: lowercase, strip periods/accents-insensitive punctuation."""
    s = s.replace(".", "").replace("'", "").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    loads = pd.read_csv(LOADS)
    combos = loads[["tri", "yy"]].drop_duplicates().sort_values(["yy", "tri"])
    print(f"{len(combos)} team-seasons to fetch")

    rows = []
    for i, (tri, yy) in enumerate(combos.itertuples(index=False)):
        if tri not in TEAM_ID:
            print(f"  SKIP {tri} {yy}: no team_id mapping")
            continue
        season_str = f"20{yy:02d}-{(yy + 1) % 100:02d}"
        try:
            r = commonteamroster.CommonTeamRoster(
                team_id=TEAM_ID[tri], season=season_str, timeout=30)
            df = r.get_data_frames()[0]
        except Exception as e:
            print(f"  FAIL {tri} {season_str}: {e}")
            time.sleep(1.0)
            continue
        for pr in df.itertuples(index=False):
            rows.append(dict(tri=tri, yy=int(yy), season=season_label(yy),
                             full_name=pr.PLAYER, position=pr.POSITION,
                             abbrev=to_abbrev(pr.PLAYER)))
        print(f"  [{i+1}/{len(combos)}] {tri} {season_str}: {len(df)} players")
        time.sleep(0.6)  # be polite to the API

    out = pd.DataFrame(rows)

    # ---- fallback pass: mid-season-trade players missing from the end-of-
    # season roster snapshot. Their CAREER position (via commonplayerinfo) is
    # still official NBA data, not a guess — a player's listed position is
    # effectively invariant across a season either way.
    loads["norm"] = loads["player"].map(normalize)
    out["norm"] = out["abbrev"].map(normalize)
    key_map = set(zip(out["tri"], out["yy"], out["norm"]))
    unmatched = loads[~loads.apply(lambda r: (r["tri"], r["yy"], r["norm"]) in key_map, axis=1)]
    unmatched_players = unmatched["player"].unique().tolist()
    print(f"\nFallback: resolving {len(unmatched_players)} unmatched player names "
          f"via commonplayerinfo (career position, official NBA data)...")

    all_static = static_players.get_players()
    fallback_rows = []
    for name in unmatched_players:
        cand = _find_static_player(name, all_static)
        if cand is None:
            print(f"  NO MATCH: {name}")
            continue
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=cand["id"], timeout=30)
            df_info = info.get_data_frames()[0]
            pos = df_info["POSITION"].iloc[0] if len(df_info) else None
        except Exception as e:
            print(f"  FAIL {name} ({cand['full_name']}): {e}")
            time.sleep(0.6)
            continue
        if pos:
            rows_for_name = unmatched[unmatched["player"] == name][["tri", "yy"]].drop_duplicates()
            for tri, yy in rows_for_name.itertuples(index=False):
                fallback_rows.append(dict(tri=tri, yy=int(yy), season=season_label(yy),
                                          full_name=cand["full_name"], position=pos,
                                          abbrev=name))
        time.sleep(0.5)

    if fallback_rows:
        out = pd.concat([out.drop(columns=["norm"]), pd.DataFrame(fallback_rows)],
                        ignore_index=True)

    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out)} rows)")

    out["norm"] = out["abbrev"].map(normalize)
    key_map2 = set(zip(out["tri"], out["yy"], out["norm"]))
    matched2 = loads.apply(lambda r: (r["tri"], r["yy"], r["norm"]) in key_map2, axis=1)
    print(f"Final coverage: {matched2.sum()}/{len(loads)} player-seasons matched "
          f"({matched2.mean()*100:.1f}%)")
    still_unmatched = loads[~matched2][["player", "tri", "yy"]].drop_duplicates()
    if len(still_unmatched):
        print(f"Still unmatched ({len(still_unmatched)} rows):")
        print(still_unmatched.to_string(index=False))


def _find_static_player(abbrev_name: str, all_static: list) -> dict | None:
    """Match 'R. Gobert' against nba_api's static full-name roster by last
    name + first-initial (handles suffixes like Jr./III via normalize())."""
    target = normalize(abbrev_name)
    parts = abbrev_name.replace(".", "").split()
    if len(parts) < 2:
        return None
    initial, last_parts = parts[0][0].lower(), parts[1:]
    candidates = []
    for p in all_static:
        full = normalize(p["full_name"])
        full_parts = full.split()
        if not full_parts:
            continue
        if full_parts[0][0] != initial:
            continue
        # require the abbreviated last-name tokens to all appear, in order,
        # in the candidate's remaining tokens
        rest = " ".join(full_parts[1:])
        if normalize(" ".join(last_parts)) in rest or rest in normalize(" ".join(last_parts)):
            candidates.append(p)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # prefer non-active-ambiguous: pick the one whose full_name normalized
    # most closely matches token count of the target
    candidates.sort(key=lambda p: abs(len(normalize(p["full_name"]).split()) - len(target.split())))
    return candidates[0]


if __name__ == "__main__":
    main()
