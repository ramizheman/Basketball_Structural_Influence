"""
Shared machinery for the Structural Role Portability study
(PORTABILITY_PREREGISTRATION.md). Builds, ONCE and cached, the per-(team-season,
player) adjacency-displacement signature ΔA_p (28-vec, off-diagonal upper triangle)
and play-type profile π_p (8-vec). Reuses Structural Load / Fragility objects with NO drift.

Signatures use cosine (pattern), so the standardization null count is a nuisance
scale param (registered N_NULL=300).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from wiring_gate import (
    PLAY_TYPES_8, IU, OUT_DIR,
    inclusion_ok, assoc_O, null_O_stats, standardize,
)
from wiring_full_study import pull_edges, split_halves
from fragility_gate import MIN_INIT_SEASON

N_NULL = 300
SEED = 17
SIG_CACHE = OUT_DIR / "portability_signatures.parquet"

SIG_COLS = [f"s{i}" for i in range(28)]
PROF_COLS = [f"p{i}" for i in range(8)]


def build_B_labeled_by_id(rows: pd.DataFrame):
    """Weighted biadjacency B keyed by the NBA person_id (player_id), NOT the
    abbreviated display name. Short names like "D. Mitchell" collide across
    real players (Donovan vs. Davion Mitchell, S. Curry Steph vs. Seth, etc.);
    player_id is the collision-safe identity. Returns (B, ids, id_to_name)."""
    piv = (rows.groupby(["player_id", "ptype"])["n"].sum()
               .unstack(fill_value=0)
               .reindex(columns=PLAY_TYPES_8, fill_value=0))
    B = piv.to_numpy(dtype=np.int64)
    mask = B.sum(axis=1) > 0
    ids = list(np.asarray(piv.index)[mask])
    B = B[mask]
    id_to_name = rows.groupby("player_id")["player"].first().to_dict()
    return B, ids, id_to_name


def loo_signatures(B, players, rng, n_null):
    """Per included player_id: ΔA off-diagonal 28-vec + play-type profile 8-vec + init/u."""
    E, sd, _ = null_O_stats(B, rng, n_null)
    A_full = standardize(assoc_O(B), E, sd)
    total = int(B.sum())
    out = {}
    for idx, p in enumerate(players):
        init = int(B[idx].sum())
        if init < MIN_INIT_SEASON:
            continue
        Bm = np.delete(B, idx, axis=0)
        if Bm.shape[0] < 2 or Bm.sum() == 0:
            continue
        Em, sdm, _ = null_O_stats(Bm, rng, n_null)
        Am = standardize(assoc_O(Bm), Em, sdm)
        dA = (A_full - Am)[IU]                    # 28-vector displacement signature
        prof = B[idx].astype(float)
        prof = prof / prof.sum() if prof.sum() > 0 else prof
        out[p] = dict(init=init, u=init / total, sig=dA, prof=prof)
    return out


def build_signature_cache(max_teams=None):
    if SIG_CACHE.exists():
        return pd.read_parquet(SIG_CACHE)
    rng = np.random.default_rng(SEED)
    df = pull_edges()
    if "player_id" not in df.columns:
        raise RuntimeError(
            "full_edges.parquet is missing player_id — delete "
            "output/full_edges.parquet and regenerate.")
    combos = (df.groupby(["team", "season_yy"])["n"].sum()
                .reset_index().sort_values(["season_yy", "team"]))
    if max_teams is not None:
        combos = combos.head(max_teams)
    rows = []
    t0 = time.time()
    print(f"\nBuilding ΔA signatures for {len(combos)} team-seasons (N_NULL={N_NULL}) ...", flush=True)
    for tri, yy, _ in combos.itertuples(index=False):
        sub = df[(df["team"] == tri) & (df["season_yy"] == yy) & (df["is_playoff"] == 0)]
        if sub.empty:
            continue
        odd, even = split_halves(sub)
        B_odd, _, _ = build_B_labeled_by_id(odd)
        B_even, _, _ = build_B_labeled_by_id(even)
        if not (inclusion_ok(B_odd) and inclusion_ok(B_even)):
            continue
        B_full, ids_full, id_to_name = build_B_labeled_by_id(sub)
        sig = loo_signatures(B_full, ids_full, rng, N_NULL)
        # Chronological anchor within team-season (orders mid-season same-yy stints)
        anchor = sub.groupby("player_id")["game_id"].min()
        for pid, s in sig.items():
            rec = dict(key=f"{tri}_{yy}", tri=tri, yy=int(yy),
                       player_id=pid, player=id_to_name.get(pid, str(pid)),
                       init=s["init"], u=s["u"], anchor=anchor.get(pid))
            rec.update({c: v for c, v in zip(SIG_COLS, s["sig"])})
            rec.update({c: v for c, v in zip(PROF_COLS, s["prof"])})
            rows.append(rec)
        print(f"  {tri}_{yy:<3} players={len(sig):>2}", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(SIG_CACHE, index=False)
    print(f"\n{len(out)} player-signatures across {out['key'].nunique()} team-seasons "
          f"in {time.time()-t0:.0f}s -> {SIG_CACHE.name}")
    return out


def cossim(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-teams", type=int, default=None)
    a = ap.parse_args()
    build_signature_cache(max_teams=a.max_teams)
