"""
WIRING — FEASIBILITY GATE
==========================

Pre-registered kill test for the flagship claim in WIRING_PREREGISTRATION.md:

    P1  NBA offenses carry a team-persistent structural identity that is NOT
        recoverable from play-type rates.

This script implements the gate EXACTLY as registered (§§3-7). It does NOT run
the full 120-team-season study. It runs the 8-team pilot and reports the three
pass/fail criteria. Any single failure => do not proceed.

Representation (§3)   : L2 player -> play-type initiation bipartite B, per team-half.
Invariant (§4)        : 8x8 play-type co-initiation ASSOCIATION matrix A, using the
                        min-overlap operator  O[i,j] = sum_player min(B[.,i], B[.,j]),
                        i != j, diagonal fixed to 0, standardized against the null.
Null (§5)             : fixed BOTH-margins token shuffle (preserves each player's
                        initiation total AND each play-type total exactly; randomizes
                        only the association). Preserves the play-type rate vector
                        exactly -> anything surviving it is "beyond the rates".
Split (§6)            : odd vs even game index within each team-season.
Inclusion (§6)        : >= 300 offensive possessions and >= 6 players with >= 20
                        initiations, per team-half.
Non-independence (§6) : franchise is the unit; BOS appears twice (one franchise).
                        Persistence permutation is season-stratified and re-pairs
                        halves within season, so franchise/era structure is respected.

Criteria (§7.1) — gate PASSES only if all three hold:
  1. clears null   : observed association deviation > 95th pct of null for >= 4/8
                     pilot team-seasons (per-team p < 0.05).
  2. persistence   : mean within-team half distance < mean between-team half distance,
                     ratio < 0.80, season-stratified permutation p < 0.05.
  3. not collinear : R^2 of pooled A cells regressed on rate features
                     (rate_i, rate_j, rate_i*rate_j) < 0.75.

Run:  python wiring_gate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------- configuration
import os
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)
NEO4J_DB = os.getenv("NEO4J_DB", "basketball")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)
EDGE_CACHE = OUT_DIR / "l2_edges.parquet"

# Registered play-type node set (8; OTHER excluded — residual catch-all). §3
PLAY_TYPES_8 = ["TRANSITION", "PNR", "DRIVE", "POST_UP", "SPOT_UP", "CUT", "PUTBACK", "PULL_UP"]
PT_INDEX = {pt: i for i, pt in enumerate(PLAY_TYPES_8)}

# Frozen pilot slate (§7.1). (tricode, season_yy, label, franchise, role)
PILOT = [
    ("SAC", 23, "Sacramento Kings 2023-24", "SAC", "distinct A (hub-big PnR/DHO)"),
    ("GSW", 23, "Golden State Warriors 2023-24", "GSW", "distinct B (motion)"),
    ("DAL", 23, "Dallas Mavericks 2023-24", "DAL", "distinct C (iso)"),
    ("DEN", 23, "Denver Nuggets 2023-24", "DEN", "look-alike to SAC (hub-big)"),
    ("BOS", 23, "Boston Celtics 2023-24", "BOS", "persistence anchor s1"),
    ("BOS", 24, "Boston Celtics 2024-25", "BOS", "persistence anchor s2"),
    ("OKC", 24, "Oklahoma City Thunder 2024-25", "OKC", "free pick"),
    ("NYK", 24, "New York Knicks 2024-25", "NYK", "free pick"),
]

INCLUDE_PLAYOFFS = False        # regular season only (roster/coach stability)
MIN_POSS_PER_HALF = 300         # §6
MIN_PLAYERS = 6                 # §6
MIN_INIT_PER_PLAYER = 20        # §6
N_NULL = 1000                   # §5 (>= 1000)
N_PERM = 5000                   # §6 (>= 5000)
SEED = 17

TRI = sorted({t for t, *_ in PILOT})
IU = np.triu_indices(8, k=1)    # 28 unique off-diagonal cells (O is symmetric)


# ------------------------------------------------------------------- data pull
def pull_edges() -> pd.DataFrame:
    if EDGE_CACHE.exists():
        print(f"Loading cached L2 edges -> {EDGE_CACHE.name}", flush=True)
        return pd.read_parquet(EDGE_CACHE)

    from neo4j import GraphDatabase

    q = """
    MATCH (pl:Player)-[:INITIATED]->(p:Possession)
    WHERE p.offensive_team_tricode IN $tris
      AND p.play_type IN $ptypes
    RETURN p.game_id                 AS game_id,
           p.offensive_team_tricode  AS team,
           pl.name                   AS player,
           p.play_type               AS ptype,
           count(*)                  AS n
    """
    t0 = time.time()
    print("Pulling L2 initiation edges from Neo4j ...", flush=True)
    drv = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with drv.session(database=NEO4J_DB) as s:
        rows = [dict(r) for r in s.run(q, tris=TRI, ptypes=PLAY_TYPES_8)]
    drv.close()

    df = pd.DataFrame(rows)
    df["game_id"] = df["game_id"].astype(str)
    df["season_yy"] = df["game_id"].str.slice(3, 5).astype(int)
    df["is_playoff"] = (df["game_id"].str.slice(0, 3) == "004").astype(int)
    df = df[df["player"].notna() & (df["player"].astype(str).str.len() > 0)]
    df.to_parquet(EDGE_CACHE, index=False)
    print(f"  {len(df):,} (game,team,player,ptype) rows in {time.time()-t0:.0f}s", flush=True)
    return df


def team_season_rows(df: pd.DataFrame, tri: str, yy: int) -> pd.DataFrame:
    m = (df["team"] == tri) & (df["season_yy"] == yy)
    if not INCLUDE_PLAYOFFS:
        m &= df["is_playoff"] == 0
    return df[m].copy()


def split_halves(rows: pd.DataFrame):
    """Odd vs even game index within the team-season (§6)."""
    games = sorted(rows["game_id"].unique())
    idx = {g: i for i, g in enumerate(games)}
    parity = rows["game_id"].map(idx) % 2
    return rows[parity == 0], rows[parity == 1]


# --------------------------------------------------------------- matrix build
def build_B(rows: pd.DataFrame):
    """Weighted biadjacency B: rows=players, cols=8 play types. Drops zero-rows."""
    piv = (rows.groupby(["player", "ptype"])["n"].sum()
               .unstack(fill_value=0)
               .reindex(columns=PLAY_TYPES_8, fill_value=0))
    B = piv.to_numpy(dtype=np.int64)
    B = B[B.sum(axis=1) > 0]
    return B


def inclusion_ok(B: np.ndarray) -> bool:
    if B.sum() < MIN_POSS_PER_HALF:
        return False
    if int((B.sum(axis=1) >= MIN_INIT_PER_PLAYER).sum()) < MIN_PLAYERS:
        return False
    return True


def assoc_O(B: np.ndarray) -> np.ndarray:
    """O[i,j] = sum_player min(B[.,i], B[.,j]); diagonal 0. Symmetric. §4"""
    O = np.zeros((8, 8))
    for i in range(8):
        bi = B[:, i]
        for j in range(i + 1, 8):
            v = np.minimum(bi, B[:, j]).sum()
            O[i, j] = O[j, i] = v
    return O


def null_O_stats(B: np.ndarray, rng: np.random.Generator, n_null: int):
    """Fixed BOTH-margins token shuffle. Returns (E, sd, O_null_stack). §5"""
    rowsums = B.sum(axis=1).astype(np.int64)
    colsums = B.sum(axis=0).astype(np.int64)
    P = len(rowsums)
    player_tokens = np.repeat(np.arange(P), rowsums)
    ptype_tokens = np.repeat(np.arange(8), colsums)

    stack = np.empty((n_null, 8, 8))
    for k in range(n_null):
        pt = ptype_tokens.copy()
        rng.shuffle(pt)
        Bn = np.zeros((P, 8), dtype=np.int64)
        np.add.at(Bn, (player_tokens, pt), 1)
        stack[k] = assoc_O(Bn)
    E = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=1)
    return E, sd, stack


def standardize(O: np.ndarray, E: np.ndarray, sd: np.ndarray) -> np.ndarray:
    A = np.zeros((8, 8))
    nz = sd > 0
    A[nz] = (O[nz] - E[nz]) / sd[nz]
    np.fill_diagonal(A, 0.0)
    return A


def dev_stat(O: np.ndarray, E: np.ndarray, sd: np.ndarray) -> float:
    """Global association deviation = sqrt(sum off-diag standardized^2)."""
    A = standardize(O, E, sd)
    return float(np.sqrt((A[IU] ** 2).sum()))


def frob_offdiag(A: np.ndarray, B: np.ndarray) -> float:
    d = A[IU] - B[IU]
    return float(np.sqrt((d ** 2).sum()))


def cos_offdiag(A: np.ndarray, B: np.ndarray) -> float:
    """Scale-invariant distance = 1 - cosine similarity on the 28-vector.
    ROBUSTNESS ONLY (not the registered §4 metric). Isolates association
    PATTERN from association MAGNITUDE, to test whether the persistence signal
    is pattern-identity or a magnitude confound."""
    a, b = A[IU], B[IU]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - (a @ b) / (na * nb))


# ------------------------------------------------------------------- pipeline
def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = pull_edges()

    per_team = {}   # label -> dict of results
    print("\n--- per team-season: build, inclusion, null (criterion 1) ---")
    for tri, yy, label, franchise, role in PILOT:
        rows = team_season_rows(df, tri, yy)
        odd, even = split_halves(rows)
        B_full = build_B(rows)
        B_odd, B_even = build_B(odd), build_B(even)

        inc = inclusion_ok(B_odd) and inclusion_ok(B_even)
        E, sd, stack = null_O_stats(B_full, rng, N_NULL)
        O_obs = assoc_O(B_full)
        D_obs = dev_stat(O_obs, E, sd)
        D_null = np.array([dev_stat(stack[k], E, sd) for k in range(N_NULL)])
        p1 = float((D_null >= D_obs).mean())
        A_team = standardize(O_obs, E, sd)

        # half-level standardized A (each half vs its own null) for persistence
        Eo, sdo, _ = null_O_stats(B_odd, rng, N_NULL)
        Ee, sde, _ = null_O_stats(B_even, rng, N_NULL)
        A_odd = standardize(assoc_O(B_odd), Eo, sdo)
        A_even = standardize(assoc_O(B_even), Ee, sde)

        rates = B_full.sum(axis=0).astype(float)
        rates = rates / rates.sum()

        per_team[label] = dict(
            tri=tri, yy=yy, franchise=franchise, role=role, incl=inc,
            n_poss=int(B_full.sum()), n_players=B_full.shape[0],
            D_obs=D_obs, D_null_mean=float(D_null.mean()),
            D_null_p95=float(np.percentile(D_null, 95)), p1=p1,
            A_team=A_team, A_odd=A_odd, A_even=A_even, rates=rates,
        )
        flag = "OK " if inc else "LOW"
        print(f"  {label:<32} [{flag}] poss={B_full.sum():>5}  "
              f"D_obs={D_obs:6.2f}  null95={np.percentile(D_null,95):6.2f}  "
              f"p={p1:.3f}", flush=True)

    labels = [l for _, _, l, _, _ in PILOT]

    # ---- CRITERION 1 --------------------------------------------------------
    n_clear = sum(per_team[l]["p1"] < 0.05 for l in labels)
    c1 = n_clear >= 4
    print(f"\nCRITERION 1 (clears null): {n_clear}/8 team-seasons p<0.05  "
          f"-> {'PASS' if c1 else 'FAIL'} (need >=4)")

    # ---- CRITERION 2 (persistence) -----------------------------------------
    # 16 halves: order [team0_odd, team0_even, team1_odd, ...]
    halves = []
    for l in labels:
        halves.append((l, per_team[l]["franchise"], per_team[l]["yy"], "odd", per_team[l]["A_odd"]))
        halves.append((l, per_team[l]["franchise"], per_team[l]["yy"], "even", per_team[l]["A_even"]))
    H = len(halves)
    season_of_half = np.array([halves[h][2] for h in range(H)])
    seasons = sorted(set(season_of_half.tolist()))
    within_idx = [(2 * k, 2 * k + 1) for k in range(len(labels))]

    def dist_matrix(fn):
        M = np.zeros((H, H))
        for a in range(H):
            for b in range(a + 1, H):
                M[a, b] = M[b, a] = fn(halves[a][4], halves[b][4])
        return M

    def between_pairs(M, team_of):
        vals = []
        for a in range(H):
            for b in range(a + 1, H):
                if halves[a][2] == halves[b][2] and team_of[a] != team_of[b]:
                    vals.append(M[a, b])
        return np.array(vals)

    def persistence(M):
        within = np.array([M[a, b] for a, b in within_idx])
        team_of_obs = [halves[h][0] for h in range(H)]
        between = between_pairs(M, team_of_obs)
        S = float(within.mean() / between.mean())
        prng = np.random.default_rng(SEED + 99)
        Sp = np.empty(N_PERM)
        for it in range(N_PERM):
            pseudo = [None] * H
            wv = []
            for sea in seasons:
                hs = [h for h in range(H) if season_of_half[h] == sea]
                prng.shuffle(hs)
                for k in range(0, len(hs), 2):
                    a, b = hs[k], hs[k + 1]
                    pseudo[a] = pseudo[b] = f"{sea}_{k}"
                    wv.append(M[a, b])
            Sp[it] = np.mean(wv) / np.mean(between_pairs(M, pseudo))
        return S, float((Sp <= S).mean()), within.mean(), between.mean()

    # registered primary: Frobenius (§4)
    Dm = dist_matrix(frob_offdiag)
    S_obs, p2, w_mean, b_mean = persistence(Dm)
    c2 = (S_obs < 0.80) and (p2 < 0.05)
    print(f"CRITERION 2 (persistence, Frobenius=registered): S={S_obs:.3f}  "
          f"perm p={p2:.4f}  -> {'PASS' if c2 else 'FAIL'} (need S<0.80 & p<0.05)")
    print(f"    within mean={w_mean:.3f}  between mean={b_mean:.3f}")

    # ROBUSTNESS (post-hoc, not registered): cosine = pattern only, magnitude removed
    Dm_cos = dist_matrix(cos_offdiag)
    S_cos, p2_cos, w_cos, b_cos = persistence(Dm_cos)
    print(f"    [robustness] cosine (pattern-only): S={S_cos:.3f}  perm p={p2_cos:.4f}  "
          f"within={w_cos:.3f} between={b_cos:.3f}")
    print(f"    [robustness] -> pattern identity {'SURVIVES' if S_cos < 0.80 and p2_cos < 0.05 else 'COLLAPSES'} "
          f"magnitude normalization")

    # ---- CRITERION 3 (collinearity with rates) -----------------------------
    X_rows, y_rows = [], []
    for l in labels:
        A = per_team[l]["A_team"]
        r = per_team[l]["rates"]
        for i, j in zip(*IU):
            X_rows.append([1.0, r[i], r[j], r[i] * r[j]])
            y_rows.append(A[i, j])
    X = np.array(X_rows)
    y = np.array(y_rows)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    c3 = r2 < 0.75
    print(f"CRITERION 3 (not collinear): R^2(A ~ rates) = {r2:.3f}  "
          f"-> {'PASS' if c3 else 'FAIL'} (need <0.75)")

    # ---- registered sanity anchors (§7.1, descriptive only) ----------------
    def team_dist(la, lb):
        return frob_offdiag(per_team[la]["A_team"], per_team[lb]["A_team"])

    def team_cos(la, lb):
        return cos_offdiag(per_team[la]["A_team"], per_team[lb]["A_team"])

    print("\n--- registered sanity anchors (descriptive, NOT confirmatory) ---")
    print("     [Frobenius=registered | cosine=pattern-only robustness]")
    for tag, la, lb, exp in [
        ("look-alike  SAC~DEN", "Sacramento Kings 2023-24", "Denver Nuggets 2023-24", "small"),
        ("persist BOS23~BOS24", "Boston Celtics 2023-24", "Boston Celtics 2024-25", "small"),
        ("distinct    SAC-GSW", "Sacramento Kings 2023-24", "Golden State Warriors 2023-24", "large"),
        ("distinct    SAC-DAL", "Sacramento Kings 2023-24", "Dallas Mavericks 2023-24", "large"),
        ("distinct    GSW-DAL", "Golden State Warriors 2023-24", "Dallas Mavericks 2023-24", "large"),
    ]:
        print(f"  {tag} (expect {exp:<5}) : frob={team_dist(la, lb):7.2f}  cos={team_cos(la, lb):.3f}")

    # ---- verdict ------------------------------------------------------------
    gate_pass = c1 and c2 and c3
    print("\n" + "=" * 66)
    print("WIRING FEASIBILITY GATE — VERDICT")
    print("=" * 66)
    print(f"  C1 clears null   : {'PASS' if c1 else 'FAIL'}  ({n_clear}/8 p<0.05)")
    print(f"  C2 persistence   : {'PASS' if c2 else 'FAIL'}  (S={S_obs:.3f}, p={p2:.4f})")
    print(f"  C3 not collinear : {'PASS' if c3 else 'FAIL'}  (R2={r2:.3f})")
    verdict = ("GATE PASSES — spine holds at pilot scale; proceed to full study"
               if gate_pass else
               "GATE FAILS — do NOT run the 120-team-season pipeline; report the negative")
    print(f"\n  {verdict}")
    print("=" * 66)

    # ---- write log ----------------------------------------------------------
    out = OUT_DIR / "gate_result.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("WIRING FEASIBILITY GATE\n")
        f.write(f"seed={SEED} N_NULL={N_NULL} N_PERM={N_PERM} "
                f"playoffs={INCLUDE_PLAYOFFS}\n\n")
        f.write("per team-season:\n")
        for l in labels:
            r = per_team[l]
            f.write(f"  {l:<32} incl={r['incl']} poss={r['n_poss']} "
                    f"players={r['n_players']} D_obs={r['D_obs']:.2f} "
                    f"p1={r['p1']:.3f}\n")
        f.write(f"\nC1 clears null   : {'PASS' if c1 else 'FAIL'} ({n_clear}/8 p<0.05)\n")
        f.write(f"C2 persistence   : {'PASS' if c2 else 'FAIL'} "
                f"S={S_obs:.3f} within={w_mean:.3f} between={b_mean:.3f} "
                f"perm_p={p2:.4f}\n")
        f.write(f"   [robustness cosine/pattern-only] S={S_cos:.3f} perm_p={p2_cos:.4f} "
                f"within={w_cos:.3f} between={b_cos:.3f}\n")
        f.write(f"C3 not collinear : {'PASS' if c3 else 'FAIL'} R2={r2:.3f}\n")
        sac, den = "Sacramento Kings 2023-24", "Denver Nuggets 2023-24"
        bos1, bos2 = "Boston Celtics 2023-24", "Boston Celtics 2024-25"
        gsw, dal = "Golden State Warriors 2023-24", "Dallas Mavericks 2023-24"
        f.write("\nanchors [frob | cos(pattern)]:\n")
        f.write(f"  SAC~DEN    {team_dist(sac, den):.2f} | {team_cos(sac, den):.3f}\n")
        f.write(f"  BOS23~BOS24 {team_dist(bos1, bos2):.2f} | {team_cos(bos1, bos2):.3f}\n")
        f.write(f"  SAC-GSW    {team_dist(sac, gsw):.2f} | {team_cos(sac, gsw):.3f}\n")
        f.write(f"  SAC-DAL    {team_dist(sac, dal):.2f} | {team_cos(sac, dal):.3f}\n")
        f.write(f"  GSW-DAL    {team_dist(gsw, dal):.2f} | {team_cos(gsw, dal):.3f}\n")
        f.write(f"\nVERDICT: {verdict}\n")
    print(f"\nWrote {out}")
    print(f"Total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
