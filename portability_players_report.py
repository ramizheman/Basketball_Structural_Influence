"""
Enriched portability player report: positions, roles, structural influence, usage.
Writes output/portability_players_enriched.csv and portability_players.html
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from wiring_gate import PLAY_TYPES_8, OUT_DIR
from fragility_tables import season_label, player_playtype_mix, ROLE
from portability_gate import build_movers, PREREG_SEASONS

DELTAS = OUT_DIR / "portability_deltas.csv"
POS = OUT_DIR / "player_positions.csv"
LOADS = OUT_DIR / "fragility_full_loads.csv"
OUT_CSV = OUT_DIR / "portability_players_enriched.csv"
OUT_HTML = OUT_DIR / "portability_players.html"


def broad(pos: str) -> str:
    if pd.isna(pos) or pos == "?":
        return "?"
    if "C" in pos and "F" not in pos:
        return "C"
    if "G" in pos and "F" not in pos:
        return "G"
    if pos in ("F", "F-G", "G-F"):
        return "F/Wing"
    if "F" in pos and "C" in pos:
        return "F/C"
    if "F" in pos:
        return "F"
    return pos


def usage_resid_series(u, L):
    u = np.asarray(u, float)
    X = np.column_stack([np.ones_like(u), u, u ** 2])
    y = np.asarray(L, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def archetype(L, u, resid):
    """Short scout label from influence + usage (exploratory, not confirmatory)."""
    if L >= 0.12 and resid > 0.02:
        return "Hidden structural hub"
    if u >= 0.18 and L >= 0.10:
        return "Primary grammar carrier"
    if u >= 0.15 and resid < -0.02:
        return "High-usage, low structural influence"
    if L < 0.04:
        return "Peripheral / replaceable wiring"
    return "Standard role occupant"


def main():
    sig = pd.read_parquet(OUT_DIR / "portability_signatures.parquet").reset_index(drop=True)
    sig = sig[sig["yy"].isin(PREREG_SEASONS)].reset_index(drop=True)
    deltas = pd.read_csv(DELTAS)
    movers, _, _ = build_movers(sig)
    # Keyed by mover_id (player_id + both stints)
    mover_meta = {m["mover_id"]: m for m in movers}

    pos_df = pd.read_csv(POS)
    pos_map = {(r.tri, int(r.yy), r.abbrev): r.position for r in pos_df.itertuples(index=False)}
    loads = pd.read_csv(LOADS, dtype={"player_id": str})
    # fit the quadratic-in-u residual (Eq. 5) on the pre-registered 2022-25 window only,
    # matching build_class_map()/load_merged(); score every row (movers can only be
    # 22-24 origin/dest here since `sig` above is already restricted, but a player's own
    # loads row is looked up by (tri, yy, player_id) below regardless).
    fit_mask = loads["yy"].isin(PREREG_SEASONS)
    loads["u_resid"] = np.nan
    loads.loc[fit_mask, "u_resid"] = usage_resid_series(
        loads.loc[fit_mask, "u"], loads.loc[fit_mask, "Lcos"])
    mix = player_playtype_mix()

    rows = []
    for r in deltas.itertuples(index=False):
        m = mover_meta.get(r.mover_id)
        if not m:
            continue
        s1 = sig.loc[m["i1"]]
        s2 = sig.loc[m["i2"]]
        tri1, yy1 = s1["tri"], int(s1["yy"])
        tri2, yy2 = s2["tri"], int(s2["yy"])
        pos1 = pos_map.get((tri1, yy1, r.player), "?")
        pos2 = pos_map.get((tri2, yy2, r.player), "?")
        # Join on player_id (abbreviated display names are not unique)
        l1 = loads[(loads.tri == tri1) & (loads.yy == yy1) & (loads.player_id == m["player_id"])]
        l2 = loads[(loads.tri == tri2) & (loads.yy == yy2) & (loads.player_id == m["player_id"])]
        x1 = mix[(mix.tri == tri1) & (mix.yy == yy1) & (mix.player == r.player)]
        x2 = mix[(mix.tri == tri2) & (mix.yy == yy2) & (mix.player == r.player)]
        L1 = float(l1["Lcos"].iloc[0]) if len(l1) else np.nan
        L2 = float(l2["Lcos"].iloc[0]) if len(l2) else np.nan
        u1 = float(l1["u"].iloc[0]) if len(l1) else float(s1["u"])
        u2 = float(l2["u"].iloc[0]) if len(l2) else float(s2["u"])
        r1 = float(l1["u_resid"].iloc[0]) if len(l1) else np.nan
        r2 = float(l2["u_resid"].iloc[0]) if len(l2) else np.nan
        role1 = x1["role"].iloc[0] if len(x1) else "?"
        role2 = x2["role"].iloc[0] if len(x2) else "?"
        top1 = x1["top1"].iloc[0] if len(x1) else "?"
        top2 = x2["top1"].iloc[0] if len(x2) else "?"
        init1 = int(s1["init"])
        init2 = int(s2["init"])
        rows.append(dict(
            player=r.player,
            player_id=m["player_id"], mover_id=m["mover_id"], gap_seasons=m["gap_seasons"],
            position=broad(pos1 if pos1 != "?" else pos2),
            position_detail=pos1 if pos1 != "?" else pos2,
            origin_team=tri1, origin_season=season_label(yy1),
            dest_team=tri2, dest_season=season_label(yy2),
            move=f"{tri1} {season_label(yy1)} → {tri2} {season_label(yy2)}",
            origin_role=role1, dest_role=role2,
            origin_top_play=top1, dest_top_play=top2,
            origin_init=init1, dest_init=init2,
            origin_usage=round(u1, 3), dest_usage=round(u2, 3),
            origin_influence=round(L1, 3), dest_influence=round(L2, 3),
            origin_influence_resid=round(r1, 3) if np.isfinite(r1) else None,
            dest_influence_resid=round(r2, 3) if np.isfinite(r2) else None,
            origin_archetype=archetype(L1, u1, r1) if np.isfinite(L1) else "?",
            dest_archetype=archetype(L2, u2, r2) if np.isfinite(L2) else "?",
            self_sim=round(r.self_sim, 3),
            decoy_sim=round(r.decoy_sim, 3),
            delta=round(r.delta, 3),
            portable="YES" if r.delta > 0 else "NO",
        ))

    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    df.to_csv(OUT_CSV, index=False)

    def table_html(sub, title):
        cols = ["player", "position", "move", "origin_role", "dest_role",
                "origin_influence", "dest_influence", "origin_usage", "dest_usage",
                "origin_archetype", "delta", "portable"]
        t = sub[cols].copy()
        return f"<h2>{title}</h2>\n{t.to_html(index=False, escape=False)}"

    top = df.head(20)
    bot = df.tail(15).sort_values("delta")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Portability — Player Results (enriched)</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
.note {{ color: #444; font-size: 0.9rem; line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; margin-bottom: 1.5rem; }}
th, td {{ border: 1px solid #ccc; padding: 5px 7px; text-align: left; vertical-align: top; }}
th {{ background: #eee; position: sticky; top: 0; }}
tr:nth-child(even) {{ background: #fafafa; }}
.yes {{ color: #0a6; font-weight: 600; }} .no {{ color: #a30; }}
</style></head><body>
<h1>Structural Role Portability — Players (n={len(df)})</h1>
<p class="note">
<b>delta</b> = how much more similar a mover's own destination wiring is vs. a play-type-matched
teammate on that team. Positive = structural role <b>travels with the player</b> (scoutable).
<b>origin_influence</b> = structural influence L(p) on the old team (Fragility metric). <b>origin_archetype</b>
is exploratory (hidden hub / grammar carrier / etc.). Positions from NBA Stats API per season.
</p>
{table_html(top, "Most portable — wiring travels (top 20)")}
{table_html(bot, "Least portable — role resets in new system (bottom 15)")}
<h2>All players (sorted by delta)</h2>
{df.to_html(index=False, escape=False)}
</body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
