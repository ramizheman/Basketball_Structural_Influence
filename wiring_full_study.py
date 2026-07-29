"""
WIRING — FULL CONFIRMATORY STUDY
=================================

Runs the confirmatory test registered in WIRING_PREREGISTRATION.md (§7.2, as
amended 2026-07-12) across ALL team-seasons, not the 8-team pilot.

Only run this AFTER the feasibility gate (wiring_gate.py) has passed. The gate
has passed (output/gate_result.txt).

Confirmatory claims (both must hold; §2):
  P1a persistence   : within-team split-half distance < between-team distance.
  P1b irreducibility: P1a survives residualizing A against the play-type rate vector.

CO-PRIMARY METRICS (Amendment 1). BOTH must clear independently:
  - Frobenius  ‖A_x − A_y‖_F   (registered §4)
  - Cosine     1 − cos(A_x,A_y) (pattern-only; magnitude removed)
P1a/P1b are confirmed IFF BOTH metrics satisfy within<between with
franchise-clustered permutation p < 0.01. A cosine pass may NOT paper over a
Frobenius miss (or vice versa); disagreement is itself the reported finding.

Non-independence (§6):
  - unit = franchise (= tricode in this 4-season window; no relocations).
  - p-value: season-stratified re-pairing permutation (a franchise's seasons are
    in separate season strata, so they are never mixed).
  - CI: cluster bootstrap resampling FRANCHISES (each brings all its team-seasons).
  LIMITATION: coach-level clustering is NOT available in this schema (no coach
  nodes); franchise clustering is the conservative available proxy. Documented
  per §10 (no data absent from the schema).

Reported quantity (Amendment 1): per-team association magnitude (D_obs) and the
magnitude-vs-pattern split.

Run:
    python wiring_full_study.py                 # full run (all team-seasons)
    python wiring_full_study.py --max-teams 12  # smoke test on a subset
    python wiring_full_study.py --n-null 500 --n-perm 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Reuse the EXACT pilot helpers so invariant (§4) and null (§5) cannot drift.
from wiring_gate import (
    PLAY_TYPES_8, IU,
    build_B, inclusion_ok, assoc_O, null_O_stats, standardize,
    frob_offdiag, cos_offdiag,
    MIN_POSS_PER_HALF, MIN_PLAYERS, MIN_INIT_PER_PLAYER,
    NEO4J_URI, NEO4J_AUTH, NEO4J_DB,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)
EDGE_CACHE = OUT_DIR / "full_edges.parquet"

INCLUDE_PLAYOFFS = False   # regular season only (matches gate; roster/coach stability)
SEED = 17
CONF_P = 0.01              # §7.2 confirmatory threshold
N_BOOT = 2000              # cluster-bootstrap resamples for the CI


# ------------------------------------------------------------------- data pull
def pull_edges() -> pd.DataFrame:
    if EDGE_CACHE.exists():
        print(f"Loading cached full-study edges -> {EDGE_CACHE.name}", flush=True)
        return pd.read_parquet(EDGE_CACHE)

    from neo4j import GraphDatabase

    q = """
    MATCH (pl:Player)-[:INITIATED]->(p:Possession)
    WHERE p.play_type IN $ptypes AND p.offensive_team_tricode IS NOT NULL
    RETURN p.game_id                 AS game_id,
           p.offensive_team_tricode  AS team,
           pl.name                   AS player,
           pl.player_id              AS player_id,
           p.play_type               AS ptype,
           count(*)                  AS n
    """
    t0 = time.time()
    print("Pulling ALL L2 initiation edges from Neo4j (this is the big pull) ...", flush=True)
    drv = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with drv.session(database=NEO4J_DB) as s:
        rows = [dict(r) for r in s.run(q, ptypes=PLAY_TYPES_8)]
    drv.close()

    df = pd.DataFrame(rows)
    df["game_id"] = df["game_id"].astype(str)
    df["season_yy"] = df["game_id"].str.slice(3, 5).astype(int)
    df["is_playoff"] = (df["game_id"].str.slice(0, 3) == "004").astype(int)
    df = df[df["player"].notna() & (df["player"].astype(str).str.len() > 0)]
    df = df[df["team"].notna() & (df["team"].astype(str).str.len() > 0)]
    # NBA person_id — unique identity key (abbreviated names are not unique)
    df["player_id"] = df["player_id"].astype(str)
    df.to_parquet(EDGE_CACHE, index=False)
    print(f"  {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)
    return df


def split_halves(rows: pd.DataFrame):
    games = sorted(rows["game_id"].unique())
    idx = {g: i for i, g in enumerate(games)}
    parity = rows["game_id"].map(idx) % 2
    return rows[parity == 0], rows[parity == 1]


# --------------------------------------------------------- build all team-halves
def build_halves(df: pd.DataFrame, rng, n_null: int, max_teams=None):
    """Returns list of half-records and a parallel team-season table.
    Each half-record: dict(key, tri, yy, half, A_raw, rates). A_resid filled later."""
    base = df[df["is_playoff"] == 0] if not INCLUDE_PLAYOFFS else df
    combos = (base.groupby(["team", "season_yy"])["n"].sum()
                  .reset_index().sort_values(["season_yy", "team"]))
    if max_teams is not None:
        combos = combos.head(max_teams)

    halves, teams = [], []
    for tri, yy, _ in combos.itertuples(index=False):
        rows = base[(base["team"] == tri) & (base["season_yy"] == yy)]
        odd, even = split_halves(rows)
        B_odd, B_even = build_B(odd), build_B(even)
        if not (inclusion_ok(B_odd) and inclusion_ok(B_even)):
            continue
        B_full = build_B(rows)
        key = f"{tri}_{yy}"

        # magnitude (D_obs) from full-season standardization (reported quantity)
        Ef, sdf, _stack = null_O_stats(B_full, rng, n_null)
        A_full = standardize(assoc_O(B_full), Ef, sdf)
        D_obs = float(np.sqrt((A_full[IU] ** 2).sum()))

        for htag, B in [("odd", B_odd), ("even", B_even)]:
            E, sd, _ = null_O_stats(B, rng, n_null)
            A = standardize(assoc_O(B), E, sd)
            rates = B.sum(axis=0).astype(float)
            rates = rates / rates.sum()
            halves.append(dict(key=key, tri=tri, yy=int(yy), half=htag,
                               A_raw=A, rates=rates))
        teams.append(dict(key=key, tri=tri, yy=int(yy),
                          n_poss=int(B_full.sum()), n_players=B_full.shape[0],
                          D_obs=D_obs))
        print(f"  built {key:<10} poss={int(B_full.sum()):>5} D_obs={D_obs:7.2f}", flush=True)
    return halves, pd.DataFrame(teams)


def residualize(halves):
    """Fill A_resid on each half: A minus OLS fit on rate features
    (1, rate_i, rate_j, rate_i*rate_j), pooled across ALL team-halves (§4)."""
    X_rows, y_rows, coords = [], [], []
    for h_idx, h in enumerate(halves):
        A, r = h["A_raw"], h["rates"]
        for i, j in zip(*IU):
            X_rows.append([1.0, r[i], r[j], r[i] * r[j]])
            y_rows.append(A[i, j])
            coords.append((h_idx, i, j))
    X, y = np.array(X_rows), np.array(y_rows)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    for h in halves:
        h["A_resid"] = np.zeros((8, 8))
    for (h_idx, i, j), rv in zip(coords, resid):
        halves[h_idx]["A_resid"][i, j] = halves[h_idx]["A_resid"][j, i] = rv
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2


# --------------------------------------------------------------- persistence
def _dist_matrix(halves, field, fn):
    H = len(halves)
    M = np.zeros((H, H))
    for a in range(H):
        Aa = halves[a][field]
        for b in range(a + 1, H):
            M[a, b] = M[b, a] = fn(Aa, halves[b][field])
    return M


def persistence(halves, M, n_perm, rng):
    """within = same team-season (odd vs even); between = different team-season,
    same season. p from season-stratified re-pairing permutation (§6). Vectorized."""
    H = len(halves)
    yy = np.array([h["yy"] for h in halves])
    key = np.array([h["key"] for h in halves])
    ut = np.triu(np.ones((H, H), bool), 1)
    same_yy = yy[:, None] == yy[None, :]
    same_key = key[:, None] == key[None, :]
    wmask0 = ut & same_key
    bmask0 = ut & (~same_key) & same_yy
    within_mean = float(M[wmask0].mean())
    between_mean = float(M[bmask0].mean())
    S = within_mean / between_mean

    seasons = sorted(set(yy.tolist()))
    season_idx = {s: [i for i in range(H) if yy[i] == s] for s in seasons}
    Sp = np.empty(n_perm)
    for it in range(n_perm):
        pseudo = np.empty(H, dtype=np.int64)
        pid = 0
        for s in seasons:
            hs = season_idx[s][:]
            rng.shuffle(hs)
            for k in range(0, len(hs) - 1, 2):
                pseudo[hs[k]] = pseudo[hs[k + 1]] = pid
                pid += 1
            if len(hs) % 2 == 1:            # defensive: leftover singleton, no within pair
                pseudo[hs[-1]] = pid
                pid += 1
        samep = pseudo[:, None] == pseudo[None, :]
        wmask = ut & samep
        bmask = ut & (~samep) & same_yy
        Sp[it] = M[wmask].mean() / M[bmask].mean()
    p = float((Sp <= S).mean())
    return S, p, within_mean, between_mean


def cluster_bootstrap_ci(halves, M, n_boot, rng):
    """Resample FRANCHISES (tricodes) with replacement; recompute S from the
    precomputed distance matrix M each time.

    Dyadic-bootstrap correctness: a franchise picked k times yields k INDEPENDENT
    copies. Within-pairs are restricted to a single copy's own two halves (so a
    copy is never paired with its duplicate → no spurious zero-distance pairs),
    and between-pairs are formed only across DISTINCT ORIGINAL franchises (so
    duplicate copies of one franchise never form a between pair). This removes
    the degeneracy that otherwise biases S downward under cluster resampling."""
    tri = np.array([h["tri"] for h in halves])
    yy = np.array([h["yy"] for h in halves])
    key = np.array([h["key"] for h in halves])
    by_fr = {}
    for idx, h in enumerate(halves):
        by_fr.setdefault(h["tri"], []).append(idx)
    franchises = list(by_fr.keys())

    Ss = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(franchises), size=len(franchises))
        sel, cid = [], []
        for c, fi in enumerate(pick):
            idxs = by_fr[franchises[fi]]
            sel.extend(idxs)
            cid.extend([c] * len(idxs))          # copy id => distinguishes duplicates
        sel = np.array(sel)
        cid = np.array(cid)
        D = M[np.ix_(sel, sel)]
        ks, ts, ys = key[sel], tri[sel], yy[sel]
        ut = np.triu(np.ones(D.shape, bool), 1)
        wmask = ut & (ks[:, None] == ks[None, :]) & (cid[:, None] == cid[None, :])
        bmask = ut & (ts[:, None] != ts[None, :]) & (ys[:, None] == ys[None, :])
        if D[wmask].size and D[bmask].size and D[bmask].mean() > 0:
            Ss.append(float(D[wmask].mean() / D[bmask].mean()))
    Ss = np.array(Ss)
    return float(np.percentile(Ss, 2.5)), float(np.percentile(Ss, 97.5))


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--max-teams", type=int, default=None,
                    help="cap enumerated team-seasons (smoke test only)")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = pull_edges()

    print("\n--- building team-halves (inclusion §6) ---")
    halves, teams = build_halves(df, rng, args.n_null, max_teams=args.max_teams)
    n_ts = len(teams)
    n_fr = teams["tri"].nunique()
    print(f"\nteam-seasons included : {n_ts}")
    print(f"distinct franchises   : {n_fr}  (effective inferential unit)")
    if n_fr < 20:
        print("  WARNING (§9): effective N < ~20 franchises — treat as underpowered.")

    r2 = residualize(halves)
    print(f"residualization R²(A ~ rates), pooled: {r2:.3f}")

    perm_rng = np.random.default_rng(SEED + 99)
    boot_rng = np.random.default_rng(SEED + 7)

    METRICS = [("frobenius", frob_offdiag), ("cosine", cos_offdiag)]
    results = {}
    for mname, fn in METRICS:
        for claim, field in [("P1a_raw", "A_raw"), ("P1b_resid", "A_resid")]:
            M = _dist_matrix(halves, field, fn)
            S, p, wm, bm = persistence(halves, M, args.n_perm, perm_rng)
            lo, hi = cluster_bootstrap_ci(halves, M, N_BOOT, boot_rng)
            results[(mname, claim)] = dict(S=S, p=p, within=wm, between=bm, ci=(lo, hi))
            print(f"  {mname:<9} {claim:<10} S={S:.3f}  p={p:.4f}  "
                  f"within={wm:.3f} between={bm:.3f}  95%CI[{lo:.3f},{hi:.3f}]")

    # confirmation logic (Amendment 1): BOTH metrics must clear BOTH claims
    def clears(claim):
        return all(results[(m, claim)]["S"] < 1.0 and results[(m, claim)]["p"] < CONF_P
                   for m, _ in METRICS)
    p1a = clears("P1a_raw")
    p1b = clears("P1b_resid")
    frob_p1a = results[("frobenius", "P1a_raw")]["p"] < CONF_P
    cos_p1a = results[("cosine", "P1a_raw")]["p"] < CONF_P
    disagree = frob_p1a != cos_p1a

    print("\n" + "=" * 68)
    print("WIRING CONFIRMATORY RESULT")
    print("=" * 68)
    print(f"  P1a persistence    (both metrics p<{CONF_P}) : {'CONFIRMED' if p1a else 'NOT confirmed'}")
    print(f"  P1b irreducibility (both metrics p<{CONF_P}) : {'CONFIRMED' if p1b else 'NOT confirmed'}")
    if disagree:
        print("  NOTE: Frobenius and cosine DISAGREE on P1a — per Amendment 1 this")
        print("        disagreement is the finding; P1 is NOT claimed confirmed.")
    print(f"\n  magnitude decomposition: D_obs mean={teams['D_obs'].mean():.1f} "
          f"range[{teams['D_obs'].min():.1f},{teams['D_obs'].max():.1f}]")
    verdict = ("P1 CONFIRMED — offenses have a rate-irreducible topological identity"
               if (p1a and p1b and not disagree) else
               "P1 NOT fully confirmed — see per-metric rows above")
    print(f"\n  {verdict}")
    print("=" * 68)

    out = OUT_DIR / "full_result.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("WIRING CONFIRMATORY RESULT\n")
        f.write(f"seed={SEED} n_null={args.n_null} n_perm={args.n_perm} "
                f"playoffs={INCLUDE_PLAYOFFS} max_teams={args.max_teams}\n")
        f.write(f"team_seasons={n_ts} franchises={n_fr} residual_R2={r2:.3f}\n\n")
        for (m, claim), r in results.items():
            f.write(f"{m:<9} {claim:<10} S={r['S']:.3f} p={r['p']:.4f} "
                    f"within={r['within']:.3f} between={r['between']:.3f} "
                    f"CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]\n")
        f.write(f"\nP1a={'CONFIRMED' if p1a else 'no'} P1b={'CONFIRMED' if p1b else 'no'} "
                f"disagree={disagree}\n")
        f.write(f"D_obs mean={teams['D_obs'].mean():.1f} "
                f"range[{teams['D_obs'].min():.1f},{teams['D_obs'].max():.1f}]\n")
        f.write(f"VERDICT: {verdict}\n")
    teams.to_csv(OUT_DIR / "wiring_full_team_magnitudes.csv", index=False)
    print(f"\nWrote {out}")
    print(f"Total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
