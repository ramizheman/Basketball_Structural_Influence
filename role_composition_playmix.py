"""
PLAY-MIX ADAPTATION BY ON-COURT ROLE COMPOSITION  (descriptive, graph-native)
=============================================================================
Question (user): given every player's role class (taxonomy) and who is actually on
the floor each possession (ON_COURT), how does the OFFENSE's play-type mix change
when the role composition changes?  I.e. when the connector sits, does the play mix
reorganize?

This is REAL (on-court, not simulated leave-one-out), possession-level (n~650k, not
23 absences), WITHIN-team (differenced per team-season), and DESCRIPTIVE (no wins /
plus-minus / outcomes).

DATA: ON_COURT exists for 2022-23, 2023-24, 2024-25 only (2025-26 missing). Exactly-5
on-court offensive possessions kept (99.7%).

TWO VALIDITY CAVEATS (stated, not hidden):
 1. Circularity: a high-USAGE class (TERMINAL SCORER, PRIMARY ORGANIZER) mechanically
    runs more of its own play types when on the floor. The CONNECTOR/HUB contrast is
    the clean headline (low usage => a mix shift reflects how OTHERS reorganize).
 2. Confound: a class being off-floor correlates with bench / garbage-time lineups.
    Within-team-season differencing removes team identity but not game state; this is
    ASSOCIATION, not proven tactical "adaptation".

Outputs (output/):
  role_composition_possessions.parquet   cached pull (game_id, yy, team, ptype, pl0..4)
  role_composition_playmix.csv           within-team mean Δ share by class
  fig_role_composition_playmix.png       connector with/without play mix + gradient
  role_composition_playmix.html          report

Run: python role_composition_playmix.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from wiring_gate import (
    NEO4J_URI, NEO4J_AUTH, NEO4J_DB, PLAY_TYPES_8, OUT_DIR,
)

POSS_CACHE = OUT_DIR / "role_composition_possessions.parquet"
POSS_CACHE_4S = OUT_DIR / "role_composition_possessions_4season.parquet"
LOADS = OUT_DIR / "fragility_full_loads.csv"
STATS = OUT_DIR / "player_traditional_stats.csv"

MIN_MPG = 15.0
MIN_GP = 30
MIN_POSS_STRATUM = 200          # per team-season, each of with / without
CLASSES_SHORT = {
    "PRIMARY ORGANIZER": "ORGANIZER",
    "CONNECTOR / HUB": "CONNECTOR",
    "TERMINAL SCORER": "TERMINAL",
    "ROLE OCCUPANT / SPECIALIST": "OCCUPANT",
}
ORDER = ["CONNECTOR", "ORGANIZER", "TERMINAL", "OCCUPANT"]


# ------------------------------------------------------------- data pull
def pull_possessions():
    if POSS_CACHE.exists():
        return pd.read_parquet(POSS_CACHE)
    from neo4j import GraphDatabase
    q = """
    MATCH (p:Possession)
    WHERE substring(p.game_id,0,3)<>'004' AND substring(p.game_id,3,2) IN ['22','23','24']
      AND p.offensive_team_tricode IS NOT NULL AND p.play_type IN $pt
    MATCH (pl:Player)-[:ON_COURT {side:'offense'}]->(p)
    WITH p, collect(pl.name) AS oncourt
    WHERE size(oncourt) = 5
    RETURN p.game_id AS game_id, p.offensive_team_tricode AS team,
           p.defensive_team_tricode AS opp, p.play_type AS ptype,
           p.initiator_player_name AS initiator,
           toFloat(p.points_scored) AS pts, p.period AS period,
           toFloat(p.score_margin) AS score_margin, p.outcome AS outcome,
           oncourt
    """
    print("Pulling possession-level on-court-5 + play_type (2022-24, pre-registered sample) ...", flush=True)
    t0 = time.time()
    drv = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with drv.session(database=NEO4J_DB) as s:
        rows = [dict(r) for r in s.run(q, pt=PLAY_TYPES_8)]
    drv.close()
    df = pd.DataFrame(rows)
    df["yy"] = df["game_id"].str.slice(3, 5).astype(int)
    oc = pd.DataFrame(df["oncourt"].tolist(), columns=[f"pl{i}" for i in range(5)])
    df = pd.concat([df.drop(columns=["oncourt"]), oc], axis=1)
    df.to_parquet(POSS_CACHE, index=False)
    print(f"  {len(df):,} possessions in {time.time()-t0:.0f}s -> {POSS_CACHE.name}", flush=True)
    return df


def pull_possessions_4season():
    """Pooled 2022–23..2025–26 possession pull (separate cache from the
    three-season confirmatory pipeline)."""
    if POSS_CACHE_4S.exists():
        return pd.read_parquet(POSS_CACHE_4S)
    from neo4j import GraphDatabase
    q = """
    MATCH (p:Possession)
    WHERE substring(p.game_id,0,3)<>'004' AND substring(p.game_id,3,2) IN ['22','23','24','25']
      AND p.offensive_team_tricode IS NOT NULL AND p.play_type IN $pt
    MATCH (pl:Player)-[:ON_COURT {side:'offense'}]->(p)
    WITH p, collect(pl.name) AS oncourt
    WHERE size(oncourt) = 5
    RETURN p.game_id AS game_id, p.offensive_team_tricode AS team,
           p.defensive_team_tricode AS opp, p.play_type AS ptype,
           p.initiator_player_name AS initiator,
           toFloat(p.points_scored) AS pts, p.period AS period,
           toFloat(p.score_margin) AS score_margin, p.outcome AS outcome,
           oncourt
    """
    print("Pulling possession-level on-court-5 + play_type (2022-26, pooled exploratory sample) ...", flush=True)
    t0 = time.time()
    drv = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with drv.session(database=NEO4J_DB) as s:
        rows = [dict(r) for r in s.run(q, pt=PLAY_TYPES_8)]
    drv.close()
    df = pd.DataFrame(rows)
    df["yy"] = df["game_id"].str.slice(3, 5).astype(int)
    oc = pd.DataFrame(df["oncourt"].tolist(), columns=[f"pl{i}" for i in range(5)])
    df = pd.concat([df.drop(columns=["oncourt"]), oc], axis=1)
    df.to_parquet(POSS_CACHE_4S, index=False)
    print(f"  {len(df):,} possessions in {time.time()-t0:.0f}s -> {POSS_CACHE_4S.name}", flush=True)
    return df


# ------------------------------------------------------------- role classes
def build_class_map(season_yy=(22, 23, 24)):
    """(tri,yy,player) -> short class.

    Matches paper.tex Eq. 5 (sec:resid) and Eq. 6 (sec:taxonomy) exactly:
      - resid(p): Lcos regressed on a QUADRATIC in the graph-native initiation
        share u, fit across all classified players (no mpg/gp restriction on
        the fit sample -- only season_yy restricts it, default: the
        pre-registered 2022-23..2024-25 sample).
      - u_med: box-score usage-rate MEDIAN over the rotation pool (mpg>=MIN_MPG,
        gp>=MIN_GP) -- a deliberately different variable from the u used in the
        residual, per Eq. 5's text, so the residual isn't mechanically
        correlated with the usage measure it's contrasted against.

    Same residual construction as fragility_orthogonality.load_merged().
    """
    loads = pd.read_csv(LOADS)
    stats = pd.read_csv(STATS)
    stats = stats.rename(columns={"abbrev": "player"})
    df = loads.merge(stats[["tri", "yy", "player", "usg_pct", "mpg", "gp"]],
                     on=["tri", "yy", "player"], how="left")
    if season_yy is not None:
        df = df[df["yy"].isin(season_yy)].copy()
    df = df.copy()

    rot = df[(df["mpg"] >= MIN_MPG) & (df["gp"] >= MIN_GP) & df["usg_pct"].notna()].copy()
    u_med = float(rot["usg_pct"].median())

    uu = df["u"].to_numpy(float)
    ll = df["Lcos"].to_numpy(float)
    ok = np.isfinite(uu) & np.isfinite(ll)
    X = np.column_stack([np.ones(int(ok.sum())), uu[ok], uu[ok] ** 2])
    beta, *_ = np.linalg.lstsq(X, ll[ok], rcond=None)
    resid = np.full(len(df), np.nan)
    resid[ok] = ll[ok] - X @ beta
    df["u_resid"] = resid

    cmap = {}
    for r in df.itertuples(index=False):
        if not np.isfinite(r.Lcos) or not np.isfinite(r.usg_pct) or not np.isfinite(r.u_resid):
            continue
        hi_u = r.usg_pct >= u_med
        hi_r = r.u_resid > 0
        full = {(True, True): "PRIMARY ORGANIZER", (False, True): "CONNECTOR / HUB",
                (True, False): "TERMINAL SCORER", (False, False): "ROLE OCCUPANT / SPECIALIST"}[(hi_u, hi_r)]
        cmap[(r.tri, int(r.yy), r.player)] = CLASSES_SHORT[full]
    return cmap, u_med, 0.0


# ------------------------------------------------------------- analysis
def entropy(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def hhi(p):
    p = np.asarray(p, float)
    return float((p ** 2).sum())


def main():
    df = pull_possessions()
    cmap, u_med, l_med = build_class_map()

    # per-possession composition: count of each class among the 5
    plcols = [f"pl{i}" for i in range(5)]
    yy = df["yy"].to_numpy()
    team = df["team"].to_numpy()
    print(f"classifying {len(df):,} possessions x5 on-court ...", flush=True)
    cls_arrays = []
    for c in plcols:
        names = df[c].to_numpy()
        cls_arrays.append(np.array([cmap.get((team[i], int(yy[i]), names[i]), "?")
                                    for i in range(len(df))], dtype=object))
    comp = {k: np.zeros(len(df), dtype=np.int8) for k in ORDER}
    n_classified = np.zeros(len(df), dtype=np.int8)
    for arr in cls_arrays:
        for k in ORDER:
            comp[k] += (arr == k)
        n_classified += (arr != "?")
    for k in ORDER:
        df[f"n_{k}"] = comp[k]
    df["n_classified"] = n_classified

    cov = float((n_classified >= 4).mean())
    print(f"possessions with >=4 of 5 on-court classified: {cov:.1%}")

    pt = df["ptype"].to_numpy()
    pt_idx = {p: i for i, p in enumerate(PLAY_TYPES_8)}
    pti = np.array([pt_idx[x] for x in pt])
    df["_ts"] = df["team"] + "_" + df["yy"].astype(str)

    def mix(mask):
        v = np.bincount(pti[mask], minlength=8).astype(float)
        return v / v.sum() if v.sum() > 0 else v

    # within-team-season with/without-class differencing
    results = {}
    for k in ORDER:
        nk = df[f"n_{k}"].to_numpy()
        deltas, ent_w, ent_wo, hhi_w, hhi_wo, nts = [], [], [], [], [], 0
        for ts, sub_idx in df.groupby("_ts").indices.items():
            sub_idx = np.asarray(sub_idx)
            with_m = sub_idx[nk[sub_idx] >= 1]
            without_m = sub_idx[nk[sub_idx] == 0]
            if len(with_m) < MIN_POSS_STRATUM or len(without_m) < MIN_POSS_STRATUM:
                continue
            mw = mix(with_m); mwo = mix(without_m)
            deltas.append(mw - mwo)
            ent_w.append(entropy(mw)); ent_wo.append(entropy(mwo))
            hhi_w.append(hhi(mw)); hhi_wo.append(hhi(mwo))
            nts += 1
        if nts == 0:
            results[k] = None
            continue
        D = np.vstack(deltas)
        results[k] = dict(
            n_ts=nts, mean_delta=D.mean(axis=0), sd_delta=D.std(axis=0, ddof=1),
            d_entropy=float(np.mean(ent_w) - np.mean(ent_wo)),
            d_hhi=float(np.mean(hhi_w) - np.mean(hhi_wo)),
        )

    # gradient (pooled): play mix by number of connectors on floor
    grad = {}
    nc = df["n_CONNECTOR"].to_numpy()
    for g in [0, 1, 2, 3]:
        m = (nc == g) if g < 3 else (nc >= 3)
        if m.sum() > 500:
            grad[g] = mix(np.where(m)[0])

    # ---- console -----------------------------------------------------------
    print(f"\nrole thresholds: usage median={u_med:.3f}  load median={l_med:.3f}")
    print("\n=== within-team-season play-mix delta (WITH >=1 of class MINUS WITHOUT) ===")
    for k in ORDER:
        r = results[k]
        if r is None:
            print(f"\n{k}: insufficient within-team strata"); continue
        print(f"\n{k}  (team-seasons with both strata: {r['n_ts']})  "
              f"Δentropy={r['d_entropy']:+.3f}  ΔHHI={r['d_hhi']:+.3f}")
        order = np.argsort(-np.abs(r["mean_delta"]))
        for i in order:
            print(f"    {PLAY_TYPES_8[i]:<12} Δshare={r['mean_delta'][i]*100:+5.2f} pp  "
                  f"(sd {r['sd_delta'][i]*100:4.2f})")

    save_outputs(results, grad, u_med, l_med, cov)


def save_outputs(results, grad, u_med, l_med, cov):
    rows = []
    for k in ORDER:
        r = results[k]
        if r is None:
            continue
        for i, pt in enumerate(PLAY_TYPES_8):
            rows.append(dict(role_class=k, play_type=pt,
                             mean_delta_pp=round(r["mean_delta"][i] * 100, 3),
                             sd_pp=round(r["sd_delta"][i] * 100, 3), n_ts=r["n_ts"]))
    pd.DataFrame(rows).to_csv(OUT_DIR / "role_composition_playmix.csv", index=False)

    # figure: connector with/without + gradient
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    r = results["CONNECTOR"]
    if r is not None:
        order = np.argsort(-r["mean_delta"])
        pts = [PLAY_TYPES_8[i] for i in order]
        vals = r["mean_delta"][order] * 100
        colors = ["#3f8fd9" if v > 0 else "#d94f4f" for v in vals]
        axes[0].barh(range(len(pts)), vals, color=colors)
        axes[0].set_yticks(range(len(pts))); axes[0].set_yticklabels(pts)
        axes[0].invert_yaxis()
        axes[0].axvline(0, color="#333", lw=0.8)
        axes[0].set_xlabel("Δ play-type share (percentage points)")
        axes[0].set_title(f"CONNECTOR on floor vs not (within team-season, n_ts={r['n_ts']})\n"
                          "blue=more with connector · clean (low-usage) contrast", fontsize=10)
    if grad:
        gk = sorted(grad)
        x = np.arange(len(PLAY_TYPES_8))
        w = 0.8 / len(gk)
        for j, g in enumerate(gk):
            lab = f"{g}+ connectors" if g == max(gk) else f"{g} connector" + ("s" if g != 1 else "")
            axes[1].bar(x + j * w, grad[g] * 100, w, label=lab)
        axes[1].set_xticks(x + 0.4 - w / 2)
        axes[1].set_xticklabels(PLAY_TYPES_8, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("play-type share (%)")
        axes[1].set_title("Pooled play mix by # connectors on floor", fontsize=10)
        axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Play-Mix Adaptation by On-Court Role Composition (descriptive; "
                 "association not causation)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "fig_role_composition_playmix.png")
    plt.close(fig)

    write_html(results, u_med, l_med, cov)
    print("\nWrote role_composition_playmix.csv, fig_role_composition_playmix.png, "
          "role_composition_playmix.html")


def write_html(results, u_med, l_med, cov):
    import base64
    img = OUT_DIR / "fig_role_composition_playmix.png"
    b64 = base64.b64encode(img.read_bytes()).decode() if img.exists() else ""
    blocks = ""
    for k in ORDER:
        r = results[k]
        if r is None:
            blocks += f"<h3>{k}</h3><p>insufficient within-team strata.</p>"; continue
        order = np.argsort(-np.abs(r["mean_delta"]))
        tr = "".join(
            f"<tr><td>{PLAY_TYPES_8[i]}</td><td>{r['mean_delta'][i]*100:+.2f}</td>"
            f"<td>{r['sd_delta'][i]*100:.2f}</td></tr>" for i in order)
        blocks += (f"<h3>{k} <span style='font-weight:400;color:#666'>"
                   f"(team-seasons n={r['n_ts']}; Δentropy={r['d_entropy']:+.3f}, "
                   f"ΔHHI={r['d_hhi']:+.3f})</span></h3>"
                   f"<table><tr><th>play type</th><th>Δ share (pp)</th><th>sd</th></tr>{tr}</table>")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Play-Mix Adaptation by On-Court Role Composition</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#111}}
h1{{font-size:1.5rem}} h2{{font-size:1.1rem;margin-top:1.8rem;border-bottom:1px solid #ddd}}
h3{{font-size:1rem;margin-top:1.2rem}} .note{{color:#444;font-size:0.9rem;line-height:1.55}}
.warn{{background:#fff6e5;border-left:4px solid #e0a33e;padding:0.6rem 1rem;font-size:0.9rem}}
table{{border-collapse:collapse;font-size:0.85rem;margin:0.5rem 0 1rem}} th,td{{border:1px solid #ccc;padding:4px 9px;text-align:left}}
th{{background:#eee}} img{{max-width:100%;border:1px solid #ddd}}</style></head><body>
<h1>Play-Mix Adaptation by On-Court Role Composition</h1>
<p class="note">For each offensive possession (2022-24, exactly-5 on-court, n&asymp;650k) we know the
role class of everyone on the floor and the play that was run. This shows how the play-type mix shifts
when a role class is present vs absent, <b>differenced within each team-season</b> (removes team style).
Descriptive. Role thresholds: usage median={u_med*100:.1f}%, load median={l_med:.3f};
possessions with &ge;4/5 on-court classified: {cov:.0%}.</p>
<div class="warn"><b>Read with two caveats.</b> (1) <b>Circularity:</b> high-usage classes (TERMINAL,
ORGANIZER) mechanically run more of their own plays when present, so their rows are partly definitional
&mdash; the <b>CONNECTOR</b> row (low usage) is the clean signal. (2) <b>Confound:</b> a class being
off-floor overlaps with bench / garbage-time lineups; within-team differencing removes team identity but
not game state. This is <b>association, not proven tactical adaptation</b>.</div>
<h2>Figure</h2>
<img src="data:image/png;base64,{b64}" alt="play mix by composition">
<h2>Within-team-season Δ play-type share (WITH &ge;1 of class &minus; WITHOUT)</h2>
{blocks}
</body></html>"""
    (OUT_DIR / "role_composition_playmix.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
