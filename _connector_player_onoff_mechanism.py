"""
Per-Shaper player-season on/off scoring + mechanism decomposition.

IDENTITY RULE (mandatory):
  Key every player-season by NBA person_id (player_id), NEVER by abbreviated
  display name. Short names collide (Jalen/Jaylin Williams, Jalen/Jeff Green,
  Steph/Seth Curry). Fragility loads and signatures are already person_id-keyed;
  this script must not reintroduce abbrev merges.

Labels:
  - raw on/off PPP gaps are DESCRIPTIVE
  - adjusted gaps (opp / period / score-margin / talent) are CONDITIONAL ASSOCIATIONS
  - not causal absence effects

Outputs -> output/:
  role_composition_possessions_by_id.parquet   (ON_COURT + player_ids cache)
  shaper_player_onoff.csv
  shaper_player_onoff_quadrants.csv
  shaper_player_mechanism_cases.csv
  shaper_player_onoff_report.md

Run: python _connector_player_onoff_mechanism.py
"""
from __future__ import annotations

import sys
import time
from math import erf, sqrt

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from wiring_gate import (
    PLAY_TYPES_8, OUT_DIR, NEO4J_URI, NEO4J_AUTH, NEO4J_DB,
    assoc_O, null_O_stats, standardize, IU,
)
from role_composition_ppp import dummies, demean_by_group, score_bucket
from coverage_common import build_talent

IDCOLS = [f"pl{i}_id" for i in range(5)]
NAMECOLS = [f"pl{i}" for i in range(5)]
POSS_ID_CACHE = OUT_DIR / "role_composition_possessions_by_id.parquet"
LOADS = OUT_DIR / "fragility_full_loads.csv"
STATS = OUT_DIR / "player_traditional_stats.csv"
SIG = OUT_DIR / "portability_signatures.parquet"

YY_TO_SEASON = {22: "2022-23", 23: "2023-24", 24: "2024-25"}
MIN_MPG, MIN_GP = 15.0, 30
MIN_POSS = 150
N_BOOT = 400
SEED = 17
N_NULL_MECH = 300


def two_sided_p(t: float) -> float:
    if not np.isfinite(t):
        return float("nan")
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))


# ---------------------------------------------------------------------------
# Collision-safe identity
# ---------------------------------------------------------------------------
def merge_loads_stats_by_id(loads: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Attach box-score stats to person_id-keyed loads without abbrev cartesian joins.

    Unique (tri,yy,abbrev) rows merge normally. Collided abbrevs are matched by
    ranking graph initiation share u against box-score usg_pct within the
    collision group (high-u <-> high-usg). This keeps both real players.
    """
    loads = loads.copy()
    loads["player_id"] = loads["player_id"].astype(str)
    loads["abbrev"] = loads["player"]
    stats = stats.copy()
    stats["abbrev"] = stats["abbrev"].astype(str)

    key = ["tri", "yy", "abbrev"]
    loads["_n_load"] = loads.groupby(key)["player_id"].transform("count")
    stats["_n_stat"] = stats.groupby(key)["full_name"].transform("count")

    uniq_l = loads[loads["_n_load"] == 1].drop(columns=["_n_load"])
    uniq_s = stats[stats["_n_stat"] == 1].drop(columns=["_n_stat"])
    out_u = uniq_l.merge(uniq_s, on=key, how="left", suffixes=("", "_nba"))

    coll_rows = []
    coll_keys = loads.loc[loads["_n_load"] > 1, key].drop_duplicates()
    for tri, yy, abbrev in coll_keys.itertuples(index=False):
        lg = loads[(loads.tri == tri) & (loads.yy == yy) & (loads.abbrev == abbrev)].copy()
        sg = stats[(stats.tri == tri) & (stats.yy == yy) & (stats.abbrev == abbrev)].copy()
        lg = lg.sort_values("u", ascending=False).reset_index(drop=True)
        sg = sg.sort_values("usg_pct", ascending=False).reset_index(drop=True)
        n = min(len(lg), len(sg))
        if len(lg) != len(sg):
            print(f"  WARN collision size mismatch {tri}{yy} {abbrev}: "
                  f"loads={len(lg)} stats={len(sg)} — pairing min={n}", flush=True)
        for i in range(n):
            row = {**lg.iloc[i].to_dict(), **{c: sg.iloc[i][c] for c in sg.columns
                                              if c not in lg.columns}}
            # overwrite shared abbrev fields from stats side where needed
            for c in ["full_name", "gp", "mpg", "usg_pct", "ppg", "apg", "net_rating",
                      "season", "ast_pct", "touches", "front_ct_touches"]:
                if c in sg.columns:
                    row[c] = sg.iloc[i][c]
            coll_rows.append(row)
        if len(lg) > n:
            for i in range(n, len(lg)):
                row = lg.iloc[i].to_dict()
                row["full_name"] = None
                coll_rows.append(row)

    out_c = pd.DataFrame(coll_rows) if coll_rows else pd.DataFrame()
    out = pd.concat([out_u, out_c], ignore_index=True)
    out = out.drop(columns=[c for c in out.columns if c.startswith("_n_")], errors="ignore")
    # hard uniqueness on person_id × team-season
    dup = out.duplicated(["tri", "yy", "player_id"], keep=False)
    if dup.any():
        raise RuntimeError(
            f"person_id uniqueness failed after merge: "
            f"{out.loc[dup, ['tri','yy','player_id','player','full_name']].to_string()}"
        )
    return out


def build_shaper_universe() -> pd.DataFrame:
    """Rebuild SHAPER list from person_id-keyed loads (Eq. 5 taxonomy).

    Do NOT read structural_role_classification.csv for identity — that table
    drops player_id and still suffers abbrev cartesian joins via load_merged.
    """
    loads = pd.read_csv(LOADS, dtype={"player_id": str})
    loads = loads[loads["yy"].isin([22, 23, 24])].copy()
    stats = pd.read_csv(STATS)
    df = merge_loads_stats_by_id(loads, stats)

    # residualize L on graph-native u (Eq. 5), all classified players
    u = df["u"].to_numpy(float)
    L = df["Lcos"].to_numpy(float)
    ok = np.isfinite(u) & np.isfinite(L)
    X = np.column_stack([np.ones(ok.sum()), u[ok], u[ok] ** 2])
    beta, *_ = np.linalg.lstsq(X, L[ok], rcond=None)
    resid = np.full(len(df), np.nan)
    resid[ok] = L[ok] - (beta[0] + beta[1] * u[ok] + beta[2] * u[ok] ** 2)
    df["influence_residual"] = resid
    df["structural_influence"] = df["Lcos"]
    df["usage"] = df["usg_pct"]
    df["season"] = df["yy"].map(YY_TO_SEASON)
    df["team"] = df["tri"]

    rot = df[(df["mpg"] >= MIN_MPG) & (df["gp"] >= MIN_GP)
             & df["usage"].notna() & df["structural_influence"].notna()].copy()
    u_med = float(rot["usage"].median())
    rot["role_classification"] = np.where(
        (rot["usage"] >= u_med) & (rot["influence_residual"] > 0), "PRIMARY ORGANIZER",
        np.where(
            (rot["usage"] < u_med) & (rot["influence_residual"] > 0), "SHAPER",
            np.where(
                (rot["usage"] >= u_med) & (rot["influence_residual"] <= 0), "TERMINAL SCORER",
                "ROLE OCCUPANT / SPECIALIST",
            ),
        ),
    )
    conn = rot[rot["role_classification"] == "SHAPER"].copy()
    print(f"rotation pool={len(rot)}  shapers={len(conn)}  "
          f"usage median cut={u_med:.4f}", flush=True)
    # sanity: collided names must remain distinct person_ids
    jw = conn[(conn.team == "OKC") & (conn.season == "2022-23") & (conn.player == "J. Williams")]
    if len(jw):
        print("OKC 2022-23 J. Williams shaper rows (must be person_id-distinct):", flush=True)
        print(jw[["player_id", "full_name", "usage", "structural_influence", "influence_residual", "u"]]
              .to_string(index=False), flush=True)
    return conn, rot


def pull_possessions_by_id(force: bool = False) -> pd.DataFrame:
    """ON_COURT possessions with person_ids on every lineup slot + initiator."""
    if POSS_ID_CACHE.exists() and not force:
        df = pd.read_parquet(POSS_ID_CACHE)
        print(f"loaded {POSS_ID_CACHE.name} ({len(df):,})", flush=True)
        return df

    from neo4j import GraphDatabase
    q = """
    MATCH (p:Possession)
    WHERE substring(p.game_id,0,3) <> '004'
      AND substring(p.game_id,3,2) IN ['22','23','24']
      AND p.offensive_team_tricode IS NOT NULL
      AND p.play_type IN $pt
    MATCH (pl:Player)-[:ON_COURT {side:'offense'}]->(p)
    WITH p, collect({name: pl.name, id: toString(pl.player_id)}) AS oc
    WHERE size(oc) = 5
    RETURN p.game_id AS game_id,
           p.offensive_team_tricode AS team,
           p.defensive_team_tricode AS opp,
           p.play_type AS ptype,
           p.initiator_player_name AS initiator,
           toString(p.initiator_player_id) AS initiator_id,
           toFloat(p.points_scored) AS pts,
           p.period AS period,
           toFloat(p.score_margin) AS score_margin,
           p.outcome AS outcome,
           [x IN oc | x.name] AS names,
           [x IN oc | x.id] AS ids
    """
    print("Pulling ON_COURT possessions WITH player_ids (Neo4j)...", flush=True)
    t0 = time.time()
    drv = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with drv.session(database=NEO4J_DB) as s:
        rows = [dict(r) for r in s.run(q, pt=PLAY_TYPES_8)]
    drv.close()
    df = pd.DataFrame(rows)
    df["yy"] = df["game_id"].str.slice(3, 5).astype(int)
    names = pd.DataFrame(df["names"].tolist(), columns=NAMECOLS)
    ids = pd.DataFrame(df["ids"].tolist(), columns=IDCOLS)
    df = pd.concat([df.drop(columns=["names", "ids"]), names, ids], axis=1)
    df.to_parquet(POSS_ID_CACHE, index=False)
    print(f"  {len(df):,} possessions in {time.time()-t0:.0f}s -> {POSS_ID_CACHE.name}", flush=True)
    return df


def player_on_mask_id(sub: pd.DataFrame, player_id: str) -> np.ndarray:
    on = np.zeros(len(sub), dtype=bool)
    pid = str(player_id)
    for c in IDCOLS:
        on |= sub[c].astype(str).to_numpy() == pid
    return on


# ---------------------------------------------------------------------------
# Estimators (unchanged science; identity-safe masks)
# ---------------------------------------------------------------------------
def bootstrap_diff(on_pts, off_pts, rng, n_boot=N_BOOT):
    if len(on_pts) < 20 or len(off_pts) < 20:
        return float("nan"), float("nan"), float("nan")
    diffs = np.empty(n_boot)
    n_on, n_off = len(on_pts), len(off_pts)
    for i in range(n_boot):
        a = on_pts[rng.integers(0, n_on, n_on)].mean()
        b = off_pts[rng.integers(0, n_off, n_off)].mean()
        diffs[i] = (a - b) * 100
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(diffs.mean()), float(lo), float(hi)


def _cluster_se_pinv(X, resid, clusters):
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in pd.Series(np.arange(len(clusters))).groupby(clusters).indices.items():
        idx = np.asarray(idx)
        s = X[idx].T @ resid[idx]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    return np.sqrt(np.maximum(np.diag(V), 0.0))


def adjusted_on_effect(sub: pd.DataFrame, on: np.ndarray) -> dict:
    nan = dict(adj_pts100=np.nan, adj_se=np.nan, adj_t=np.nan, adj_p=np.nan,
               adj_ci_lo=np.nan, adj_ci_hi=np.nan)
    if on.sum() < MIN_POSS or (~on).sum() < MIN_POSS:
        return nan
    try:
        d = sub.copy()
        d["on"] = on.astype(float)
        d["sb"] = score_bucket(d["score_margin"].fillna(0).to_numpy()).astype(str)
        d["per"] = d["period"].fillna(0).astype(int).astype(str)
        ctrl = dummies(d, ["opp", "per", "sb"])
        X = pd.concat([d[["on", "TALENT"]].astype(float), ctrl], axis=1)
        keep = [c for c in X.columns if c == "on" or float(X[c].std()) > 1e-12]
        X = X[keep]
        y = d["pts"].to_numpy(float)
        games = d["game_id"].to_numpy()
        yb = y - y.mean()
        Xb = X.to_numpy(float)
        Xb = Xb - Xb.mean(axis=0, keepdims=True)
        if np.std(Xb[:, 0]) < 1e-12:
            return nan
        beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        resid = yb - Xb @ beta
        se = _cluster_se_pinv(Xb, resid, games)
        b, s = float(beta[0]), float(se[0])
        t = b / s if s > 0 else float("nan")
        return dict(
            adj_pts100=b * 100, adj_se=s * 100, adj_t=t, adj_p=two_sided_p(t),
            adj_ci_lo=(b - 1.96 * s) * 100, adj_ci_hi=(b + 1.96 * s) * 100,
        )
    except Exception as e:
        print(f"  adj fail: {e}", flush=True)
        return nan


def ptype_shares(pts_df):
    if len(pts_df) == 0:
        return {pt: np.nan for pt in PLAY_TYPES_8}
    vc = pts_df["ptype"].value_counts(normalize=True)
    return {pt: float(vc.get(pt, 0.0)) for pt in PLAY_TYPES_8}


def ptype_ppp(pts_df):
    out = {}
    for pt in PLAY_TYPES_8:
        sub = pts_df.loc[pts_df["ptype"] == pt, "pts"]
        out[pt] = float(sub.mean()) if len(sub) >= 30 else float("nan")
    return out


def top_initiators_by_id(pts_df, ptype, id_to_name, k=5):
    sub = pts_df[pts_df["ptype"] == ptype]
    if sub.empty:
        return []
    g = sub.groupby("initiator_id").size().sort_values(ascending=False)
    total = int(g.sum())
    rows = []
    for pid, n in g.head(k).items():
        if pd.isna(pid) or str(pid) in ("", "None", "nan"):
            continue
        name = id_to_name.get(str(pid), str(pid))
        rows.append((name, str(pid), float(n) / total, int(n)))
    return rows


def build_A_from_poss(poss, rng, n_null=N_NULL_MECH):
    v = poss[poss["initiator_id"].notna() & (poss["initiator_id"].astype(str) != "")].copy()
    if v.empty:
        return None
    piv = (v.groupby(["initiator_id", "ptype"]).size()
             .unstack(fill_value=0)
             .reindex(columns=PLAY_TYPES_8, fill_value=0))
    B = piv.to_numpy(dtype=np.int64)
    B = B[B.sum(axis=1) > 0]
    if B.shape[0] < 3 or B.sum() < 200:
        return None
    E, sd, _ = null_O_stats(B, rng, n_null)
    return standardize(assoc_O(B), E, sd)


def deltaA_summary(dA, k=3):
    pairs = []
    for i in range(8):
        for j in range(i + 1, 8):
            pairs.append((abs(dA[i, j]), dA[i, j], PLAY_TYPES_8[i], PLAY_TYPES_8[j]))
    pairs.sort(reverse=True)
    return "; ".join(f"{a}-{b}:{v:+.1f}" for _, v, a, b in pairs[:k])


def da_vec_summary(row):
    cols = [f"s{i}" for i in range(28)]
    if any(c not in row.index for c in cols):
        return ""
    v = row[cols].to_numpy(float)
    pairs, idx = [], 0
    for i in range(8):
        for j in range(i + 1, 8):
            pairs.append((abs(v[idx]), v[idx], PLAY_TYPES_8[i], PLAY_TYPES_8[j]))
            idx += 1
    pairs.sort(reverse=True)
    return "; ".join(f"{a}-{b}:{val:+.2f}" for _, val, a, b in pairs[:3])


def mechanism_block(sub, on, player_id, player_name, rng):
    on_df, off_df = sub.loc[on], sub.loc[~on]
    shares_on, shares_off = ptype_shares(on_df), ptype_shares(off_df)
    ppp_on, ppp_off = ptype_ppp(on_df), ptype_ppp(off_df)

    # name map for this team-season slice
    id_to_name = {}
    for i in range(5):
        for pid, nm in zip(sub[f"pl{i}_id"].astype(str), sub[f"pl{i}"].astype(str)):
            id_to_name[pid] = nm
    id_to_name[str(player_id)] = player_name

    own = on_df[on_df["initiator_id"].astype(str) == str(player_id)]
    own_diet = ptype_shares(own) if len(own) else {pt: 0.0 for pt in PLAY_TYPES_8}
    top_own = sorted(own_diet.items(), key=lambda x: -x[1])[:3]
    top_ptypes = [pt for pt, s in top_own if s > 0]

    init_shift = {}
    for pt in top_ptypes:
        init_shift[pt] = {
            "on_top": top_initiators_by_id(on_df, pt, id_to_name, 4),
            "off_top": top_initiators_by_id(off_df, pt, id_to_name, 4),
            "share_on": shares_on[pt], "share_off": shares_off[pt],
            "ppp_on": ppp_on[pt], "ppp_off": ppp_off[pt],
        }

    A_on, A_off = build_A_from_poss(on_df, rng), build_A_from_poss(off_df, rng)
    onoff_disp, onoff_top = float("nan"), ""
    if A_on is not None and A_off is not None:
        dA = A_on - A_off
        v_on, v_off = A_on[IU], A_off[IU]
        denom = np.linalg.norm(v_on) * np.linalg.norm(v_off)
        onoff_disp = 1.0 - float(np.dot(v_on, v_off) / denom) if denom > 0 else float("nan")
        onoff_top = deltaA_summary(dA, 4)

    mix_delta = {pt: shares_on[pt] - shares_off[pt] for pt in PLAY_TYPES_8}
    top_mix = sorted(mix_delta.items(), key=lambda x: -abs(x[1]))[:4]
    return dict(
        own_diet="; ".join(f"{pt}:{s:.2f}" for pt, s in top_own),
        top_mix_delta="; ".join(f"{pt}:{d:+.3f}" for pt, d in top_mix),
        onoff_struct_L=onoff_disp, onoff_struct_top=onoff_top,
        init_shift=init_shift, shares_on=shares_on, shares_off=shares_off,
        ppp_on=ppp_on, ppp_off=ppp_off,
    )


def build_talent_by_id(df):
    """TALENT control using initiator_id when available; falls back to name builder."""
    # coverage_common.build_talent keys on initiator name — fine for talent mean
    # of on-court five, but on-court five must be identified by ID for presence.
    # Reuse existing name-based talent (same leave-game-out definition).
    tmp = df.copy()
    if "initiator" not in tmp.columns:
        tmp["initiator"] = tmp.get("initiator_id")
    return build_talent(tmp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    conn, rot = build_shaper_universe()

    # contrasts: exact (player_id, team, season) — not all seasons for that person
    loads = pd.read_csv(LOADS, dtype={"player_id": str})
    contrast_specs = [
        ("K. Towns", "MIN", 23),
        ("L. James", "LAL", 23),
        ("J. Tatum", "BOS", 22),
    ]
    contrast_keys = []
    for name, tri, yy in contrast_specs:
        hit = loads[(loads.player == name) & (loads.tri == tri) & (loads.yy == yy)]
        if len(hit) != 1:
            print(f"  WARN contrast {name} {tri}{yy}: {len(hit)} load rows", flush=True)
            continue
        contrast_keys.append((str(hit.iloc[0]["player_id"]), tri, int(yy)))
    cmask = [
        (str(pid), tri, int(yy)) in contrast_keys
        for pid, tri, yy in zip(rot["player_id"], rot["tri"], rot["yy"])
    ]
    contrasts = rot.loc[cmask].copy()
    contrasts["is_contrast"] = True
    conn = conn.copy()
    conn["is_contrast"] = False
    universe = pd.concat([conn, contrasts], ignore_index=True)
    # person_id may repeat across team-seasons (trades); the atomic key is
    # (player_id, team, season).
    dup = universe.duplicated(["player_id", "team", "season"], keep=False)
    if dup.any():
        raise RuntimeError(
            "duplicate (player_id, team, season) in universe:\n"
            + universe.loc[dup, ["player_id", "player", "team", "season"]].to_string()
        )

    print(f"universe: {(~universe.is_contrast).sum()} shapers + "
          f"{universe.is_contrast.sum()} contrasts", flush=True)

    df = pull_possessions_by_id()
    df = df[df["pts"].notna()].copy()
    for c in IDCOLS:
        df[c] = df[c].astype(str)
    df["initiator_id"] = df["initiator_id"].astype(str)

    print("building TALENT control...", flush=True)
    df["TALENT"] = build_talent_by_id(df)

    sig = pd.read_parquet(SIG) if SIG.exists() else pd.DataFrame()
    if len(sig):
        sig["player_id"] = sig["player_id"].astype(str)

    rng = np.random.default_rng(SEED)
    ts_cache = {}
    rows = []

    for _, r in universe.iterrows():
        player_id = str(r["player_id"])
        player, team, season = r["player"], r["team"], r["season"]
        yy = int(r["yy"])
        key = (team, yy)
        if key not in ts_cache:
            ts_cache[key] = df[(df["team"] == team) & (df["yy"].astype(int) == yy)].copy()
        sub = ts_cache[key]
        on = player_on_mask_id(sub, player_id)
        n_on, n_off = int(on.sum()), int((~on).sum())
        on_pts = sub.loc[on, "pts"].to_numpy(float)
        off_pts = sub.loc[~on, "pts"].to_numpy(float)
        on_ppp = float(on_pts.mean()) if n_on else float("nan")
        off_ppp = float(off_pts.mean()) if n_off else float("nan")
        raw = (on_ppp - off_ppp) * 100 if n_on and n_off else float("nan")
        boot_mean, boot_lo, boot_hi = bootstrap_diff(on_pts, off_pts, rng)
        adj = adjusted_on_effect(sub, on)

        da_sum, sig_u = "", float("nan")
        if len(sig):
            hit = sig[(sig.player_id == player_id) & (sig.tri == team) & (sig.yy == yy)]
            if not hit.empty:
                da_sum = da_vec_summary(hit.iloc[0])
                sig_u = float(hit.iloc[0]["u"])

        rows.append(dict(
            player_id=player_id,
            player=player,
            full_name=r.get("full_name", ""),
            team=team,
            season=season,
            team_season=f"{team} {season}",
            structural_class=r["role_classification"],
            is_contrast=bool(r["is_contrast"]),
            usage=float(r["usage"]),
            structural_influence=float(r["structural_influence"]),
            influence_residual=float(r["influence_residual"]),
            graph_u=float(r["u"]) if np.isfinite(r["u"]) else sig_u,
            n_on=n_on, n_off=n_off,
            on_ppp=on_ppp, off_ppp=off_ppp,
            raw_pts100=raw,
            boot_pts100=boot_mean, boot_ci_lo=boot_lo, boot_ci_hi=boot_hi,
            adj_pts100=adj["adj_pts100"], adj_se=adj["adj_se"],
            adj_t=adj["adj_t"], adj_p=adj["adj_p"],
            adj_ci_lo=adj["adj_ci_lo"], adj_ci_hi=adj["adj_ci_hi"],
            deltaA_top=da_sum,
        ))
        if len(rows) % 20 == 0:
            print(f"  processed {len(rows)}/{len(universe)}", flush=True)

    out = pd.DataFrame(rows)
    assert not out.duplicated(["player_id", "team", "season"]).any()

    conn_out = out[~out["is_contrast"]].copy()
    conn_out["rank_raw"] = conn_out["raw_pts100"].rank(ascending=False, method="min")
    conn_out["rank_adj"] = conn_out["adj_pts100"].rank(ascending=False, method="min")
    L_med = float(conn_out["structural_influence"].median())
    S_med = float(conn_out["adj_pts100"].median())

    def quad(row):
        hi_L = row["structural_influence"] >= L_med
        hi_S = row["adj_pts100"] >= S_med if np.isfinite(row["adj_pts100"]) else False
        if hi_L and hi_S:
            return "A_highL_highScore"
        if hi_L and not hi_S:
            return "B_highL_lowScore"
        if (not hi_L) and hi_S:
            return "C_lowL_highScore"
        return "D_lowL_lowScore"

    conn_out["quadrant"] = conn_out.apply(quad, axis=1)
    out = out.merge(
        conn_out[["player_id", "team", "season", "rank_raw", "rank_adj", "quadrant"]],
        on=["player_id", "team", "season"], how="left",
    )
    out_path = OUT_DIR / "shaper_player_onoff.csv"
    out.sort_values(["is_contrast", "adj_pts100"], ascending=[True, False]).to_csv(
        out_path, index=False)
    print(f"\nwrote {out_path} ({len(out)} rows, "
          f"{out.player_id.nunique()} unique player_ids)", flush=True)

    qtab = (conn_out.groupby("quadrant")
            .agg(n=("player_id", "size"),
                 mean_L=("structural_influence", "mean"),
                 mean_r=("influence_residual", "mean"),
                 mean_usage=("usage", "mean"),
                 mean_raw=("raw_pts100", "mean"),
                 mean_adj=("adj_pts100", "mean"),
                 median_adj=("adj_pts100", "median"))
            .reset_index())
    qtab.to_csv(OUT_DIR / "shaper_player_onoff_quadrants.csv", index=False)
    print(f"\nQuadrant cuts: L median={L_med:.3f}, adj pts/100 median={S_med:.2f}")
    print(qtab.to_string(index=False))

    # mechanism cases
    ok = conn_out[(conn_out.n_on >= MIN_POSS) & (conn_out.n_off >= MIN_POSS)]
    top = ok.nlargest(5, "adj_pts100")
    bot = ok.nsmallest(5, "adj_pts100")
    gobert_23 = loads.loc[
        (loads.player == "R. Gobert") & (loads.tri == "MIN") & (loads.yy == 23), "player_id"
    ].astype(str)
    specials = out[
        ((out.player_id.astype(str).isin(gobert_23)) & (out.team == "MIN") & (out.season == "2023-24"))
        | (out.is_contrast)
        | ((out.player == "D. Gafford") & (out.season == "2022-23"))
    ]
    case_keys = pd.concat([top, bot, specials]).drop_duplicates(
        subset=["player_id", "team", "season"]
    )

    mech_rows = []
    print("\n=== MECHANISM CASES ===", flush=True)
    for _, r in case_keys.iterrows():
        yy = {v: k for k, v in YY_TO_SEASON.items()}[r["season"]]
        sub = ts_cache[(r["team"], yy)]
        on = player_on_mask_id(sub, str(r["player_id"]))
        mech = mechanism_block(sub, on, r["player_id"], r["player"], rng)
        label = r.get("full_name") or r["player"]
        print(f"\n{label} [{r['player_id']}] | {r['team']} {r['season']} | {r['structural_class']}")
        print(f"  L={r['structural_influence']:.3f} r={r['influence_residual']:+.3f} u={r['usage']:.3f}")
        print(f"  raw={r['raw_pts100']:+.1f} adj={r['adj_pts100']:+.1f} "
              f"[{r['adj_ci_lo']:+.1f},{r['adj_ci_hi']:+.1f}] on={r['n_on']} off={r['n_off']}")
        print(f"  own diet: {mech['own_diet']}")
        print(f"  mix Δ: {mech['top_mix_delta']}")
        print(f"  on/off struct L={mech['onoff_struct_L']:.3f}  top={mech['onoff_struct_top']}")
        for pt, info in mech["init_shift"].items():
            on_s = ", ".join(f"{n}({s:.0%})" for n, _, s, _ in info["on_top"][:3])
            off_s = ", ".join(f"{n}({s:.0%})" for n, _, s, _ in info["off_top"][:3])
            print(f"  {pt}: share {info['share_on']:.3f}->{info['share_off']:.3f} "
                  f"PPP {info['ppp_on']:.3f}->{info['ppp_off']:.3f}")
            print(f"      on:  {on_s}")
            print(f"      off: {off_s}")
        flat = dict(
            player_id=r["player_id"], player=r["player"], full_name=label,
            team=r["team"], season=r["season"],
            structural_class=r["structural_class"],
            structural_influence=r["structural_influence"], influence_residual=r["influence_residual"],
            usage=r["usage"], raw_pts100=r["raw_pts100"], adj_pts100=r["adj_pts100"],
            quadrant=r.get("quadrant", ""),
            own_diet=mech["own_diet"], top_mix_delta=mech["top_mix_delta"],
            onoff_struct_L=mech["onoff_struct_L"],
            onoff_struct_top=mech["onoff_struct_top"],
            loo_deltaA_top=r.get("deltaA_top", ""),
        )
        for pt in PLAY_TYPES_8:
            flat[f"share_on_{pt}"] = mech["shares_on"][pt]
            flat[f"share_off_{pt}"] = mech["shares_off"][pt]
            flat[f"ppp_on_{pt}"] = mech["ppp_on"][pt]
            flat[f"ppp_off_{pt}"] = mech["ppp_off"][pt]
        for pt, info in mech["init_shift"].items():
            flat[f"init_on_{pt}"] = "; ".join(
                f"{n}:{pid}:{s:.2f}" for n, pid, s, _ in info["on_top"][:3])
            flat[f"init_off_{pt}"] = "; ".join(
                f"{n}:{pid}:{s:.2f}" for n, pid, s, _ in info["off_top"][:3])
        mech_rows.append(flat)

    mech_df = pd.DataFrame(mech_rows)
    mech_df.to_csv(OUT_DIR / "shaper_player_mechanism_cases.csv", index=False)

    report = build_report(out, conn_out, qtab, mech_df, L_med, S_med)
    (OUT_DIR / "shaper_player_onoff_report.md").write_text(report, encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


def build_report(out, conn_out, qtab, mech_df, L_med, S_med):
    ok = conn_out[(conn_out.n_on >= MIN_POSS) & (conn_out.n_off >= MIN_POSS)]
    r_raw = ok["structural_influence"].corr(ok["raw_pts100"])
    r_adj = ok["structural_influence"].corr(ok["adj_pts100"])
    r_res_adj = ok["influence_residual"].corr(ok["adj_pts100"])
    top10 = ok.nlargest(10, "adj_pts100")
    bot10 = ok.nsmallest(10, "adj_pts100")
    resilient = ok[(ok.structural_influence >= L_med) & (ok.adj_pts100 <= S_med)].nsmallest(10, "adj_pts100")
    contrasts = out[out.is_contrast]

    def fmt(df):
        cols = ["player_id", "full_name", "player", "team_season", "structural_influence",
                "influence_residual", "usage", "raw_pts100", "adj_pts100",
                "adj_ci_lo", "adj_ci_hi", "n_on", "n_off", "quadrant"]
        use = [c for c in cols if c in df.columns]
        try:
            return df[use].to_markdown(index=False, floatfmt=".3f")
        except Exception:
            return df[use].to_string(index=False)

    lines = [
        "# Shaper Player-Season On/Off Scoring — Full Analysis",
        "",
        "**Identity:** every row keyed by `player_id` (NBA person_id). "
        "ON_COURT presence uses `pl*_id`. Abbreviated names are display-only.",
        "",
        "**Status:** exploratory / descriptive. Raw = unadjusted on vs off offensive PPP. "
        "Adjusted = within-team-season association controlling for opponent, period, "
        "score-margin bucket, and leave-game-out lineup TALENT. Not causal.",
        "",
        f"- Shapers analyzed: **{len(conn_out)}**",
        f"- Adequate sample (≥{MIN_POSS} on and off): **{len(ok)}**",
        f"- Quadrant cuts: median L = **{L_med:.3f}**, median adj pts/100 = **{S_med:.2f}**",
        f"- Corr(L, raw) = **{r_raw:+.3f}**; Corr(L, adj) = **{r_adj:+.3f}**; "
        f"Corr(r, adj) = **{r_res_adj:+.3f}**",
        "",
        "## Quadrants",
        "",
    ]
    try:
        lines.append(qtab.to_markdown(index=False, floatfmt=".3f"))
    except Exception:
        lines.append(qtab.to_string(index=False))
    lines += [
        "",
        "## 1. Highest scoring-impact shapers (adj pts/100)",
        "", fmt(top10), "",
        "## 2. High-L scoring-resilient shapers",
        "", fmt(resilient), "",
        "## 3. Terminal contrasts (low structural influence)",
        "", fmt(contrasts), "",
        "## Lowest adjusted-impact shapers",
        "", fmt(bot10), "",
        "## Mechanism cases",
        "",
    ]
    if len(mech_df):
        slim = mech_df[["player_id", "full_name", "team", "season", "structural_influence",
                        "adj_pts100", "own_diet", "top_mix_delta", "onoff_struct_L"]]
        try:
            lines.append(slim.to_markdown(index=False, floatfmt=".3f"))
        except Exception:
            lines.append(slim.to_string(index=False))
    lines += [
        "",
        "## Full ranked shaper table",
        "",
        fmt(ok.sort_values("adj_pts100", ascending=False)),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
