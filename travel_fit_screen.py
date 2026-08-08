"""
TRAVEL x FIT decision screen (pre-move, honest out-of-sample).

Two orthogonal axes a front office needs BEFORE a signing:

  TRAVEL (property of the player) -- will his organizational function reproduce
    on a new team?  FORECAST from origin-only history:
      origin structural influence, year-to-year signature stability, origin usage.
    Validated leave-one-out (closed-form OLS): the model never sees a player's
    own row when predicting it, and never sees any destination data.

  FIT (property of the match) -- does the destination LACK that function?
    MEASURED (not forecast) from the destination's existing roster: decoy_sim is
    the similarity between the mover's origin signature and a profile-matched
    incumbent already on the destination.  Low decoy_sim => structural hole => fit.
    Observable before the mover arrives.

A good acquisition needs BOTH: a function that travels AND a hole to fill.

Outputs:
  output/travel_fit_screen.csv
Run: python travel_fit_screen.py
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

OUT_DIR = Path(__file__).resolve().parent / "output"
SIG_COLS = [f"s{i}" for i in range(28)]
PROF_COLS = [f"p{i}" for i in range(8)]
K_BLOCKER_POOL = 10  # profile-plausible candidate-blocker pool (wider than the K=5
                     # decoy_sim used by the locked T1 test, so a real high-minutes
                     # blocker isn't excluded just outside the top-5-by-profile cut)


def cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def crowding_features(d, sig, stats):
    """Second FIT axis: is the destination's structural hole actually occupied by
    someone who PLAYS, not just someone who is stylistically similar on paper?

    decoy_sim (existing, mean of K=5 nearest-by-profile) answers "does the
    destination roster already contain this function" -- a pure similarity
    question, blind to whether that incumbent gets minutes. It is kept
    unchanged as the FUNCTIONAL fit axis (drives fit_score, unchanged).

    Here we add an OPPORTUNITY / crowding axis: among a wider profile-plausible
    pool (K_BLOCKER_POOL), weight structural similarity by the incumbent's
    real minutes share, and separately track the single most-similar
    high-minutes blocker. A stylistic twin buried on the bench isn't really
    occupying the mover's role; a real blocker is one who both looks like him
    AND plays."""
    stats_dedup = (stats.drop_duplicates(subset=["abbrev", "tri", "yy"], keep="first")
                        .set_index(["abbrev", "tri", "yy"])["mpg"])

    wtd_list, max_list, max_mpg_list, n_pool_list = [], [], [], []
    for r in d.itertuples():
        dest_yy = int(str(r.dest_season).split("-")[0][2:])
        dest_key = f"{r.dest_team}_{dest_yy}"
        origin_row = sig[(sig["player_id"] == r.player_id) & (sig["tri"] == r.origin_team)]
        cohort = sig[(sig["key"] == dest_key) & (sig["player_id"] != r.player_id)]
        if origin_row.empty or cohort.empty:
            wtd_list.append(np.nan); max_list.append(np.nan)
            max_mpg_list.append(np.nan); n_pool_list.append(0)
            continue
        orow = origin_row.iloc[0]
        a_sig = orow[SIG_COLS].to_numpy(float)
        a_prof = orow[PROF_COLS].to_numpy(float)

        cohort = cohort.copy()
        cohort["prof_sim"] = [cos(a_prof, cohort.loc[i, PROF_COLS].to_numpy(float))
                              for i in cohort.index]
        cohort["sig_sim"] = [cos(a_sig, cohort.loc[i, SIG_COLS].to_numpy(float))
                             for i in cohort.index]
        pool = cohort.sort_values("prof_sim", ascending=False).head(K_BLOCKER_POOL).copy()

        mpg = []
        for nm in pool["player"]:
            try:
                mpg.append(float(stats_dedup.loc[(nm, r.dest_team, dest_yy)]))
            except KeyError:
                mpg.append(0.0)
        pool["mpg"] = mpg
        n_pool_list.append(len(pool))

        w = pool["mpg"].to_numpy(float)
        if w.sum() > 0:
            wtd_list.append(float(np.average(pool["sig_sim"].to_numpy(float), weights=w)))
        else:
            wtd_list.append(float(pool["sig_sim"].mean()) if len(pool) else np.nan)

        max_list.append(float(pool["sig_sim"].max()) if len(pool) else np.nan)
        if len(pool):
            top_blocker = pool.loc[pool["mpg"].idxmax()]
            max_mpg_list.append(float(top_blocker["mpg"]))
        else:
            max_mpg_list.append(np.nan)

    return (np.array(wtd_list), np.array(max_list),
            np.array(max_mpg_list), np.array(n_pool_list))


def origin_stability(sig, player_id, origin_tri):
    """Year-to-year signature self-similarity on the ORIGIN team (pre-move, origin-only).
    Keyed by player_id, not display name (names collide across real players)."""
    g = sig[(sig["player_id"] == player_id) & (sig["tri"] == origin_tri)]
    if len(g) < 2:
        return np.nan
    vs = [g.iloc[i][SIG_COLS].to_numpy(float) for i in range(len(g))]
    cs = [cos(vs[i], vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
    return float(np.mean(cs)) if cs else np.nan


def zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def loo_ols_predictions(X, y):
    """Exact leave-one-out predictions for OLS via the hat matrix.
    pred_i (without obs i) = y_i - resid_i / (1 - h_ii)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    yhat = X @ beta
    H = X @ XtX_inv @ X.T
    h = np.clip(np.diag(H), 0, 0.999999)
    resid = y - yhat
    loo = y - resid / (1 - h)
    return loo, beta


def main():
    enr = pd.read_csv(OUT_DIR / "portability_players_enriched.csv", dtype={"player_id": str})
    sig = pd.read_parquet(OUT_DIR / "portability_signatures.parquet")
    sig["player_id"] = sig["player_id"].astype(str)
    sig = sig[sig["yy"].isin((22, 23, 24))].reset_index(drop=True)  # pre-registered window
    d = enr.dropna(subset=["self_sim", "decoy_sim", "delta", "origin_influence",
                           "origin_usage"]).copy().reset_index(drop=True)

    # origin-team tricode (first token of "origin_team", already a tricode)
    d["origin_tri"] = d["origin_team"].astype(str).str.strip()
    d["stability"] = [origin_stability(sig, pid, t)
                      for pid, t in zip(d["player_id"], d["origin_tri"])]
    # impute missing stability (single-season origins) with the median
    d["stability"] = d["stability"].fillna(d["stability"].median())

    # ---- TRAVEL: forecast self_sim from ORIGIN-ONLY features, leave-one-out ----
    feats = ["origin_influence", "stability", "origin_usage"]
    Xz = np.column_stack([np.ones(len(d))] + [zscore(d[f]) for f in feats])
    y = d["self_sim"].to_numpy(float)
    loo, beta = loo_ols_predictions(Xz, y)
    d["travel_score"] = loo  # out-of-sample predicted self_sim

    # OOS validation of the travel axis
    travelled = (d["self_sim"] > d["self_sim"].median()).to_numpy()
    pred_hi = (d["travel_score"] > np.median(d["travel_score"])).to_numpy()
    acc = float((pred_hi == travelled).mean())
    r_oos = float(np.corrcoef(d["travel_score"], d["self_sim"])[0, 1])
    # rank AUC (Mann-Whitney) for predicting travelled
    from itertools import product
    pos = d["travel_score"][travelled].to_numpy()
    neg = d["travel_score"][~travelled].to_numpy()
    auc = float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0
                         for a, b in product(pos, neg)]))

    # ---- FIT: measured hole in the destination (pre-move) ----
    d["fit_score"] = -d["decoy_sim"]  # low decoy => hole => fit  (FUNCTIONAL fit, unchanged)

    # ---- CROWDING: is that hole actually occupied by someone who plays? ----
    stats = pd.read_csv(OUT_DIR / "player_traditional_stats.csv")
    decoy_sim_wtd, decoy_sim_max, blocker_mpg, n_pool = crowding_features(d, sig, stats)
    d["decoy_sim_wtd"] = decoy_sim_wtd
    d["decoy_sim_max"] = decoy_sim_max
    d["blocker_mpg"] = blocker_mpg
    d["n_blocker_pool"] = n_pool
    d["crowding_score"] = -d["decoy_sim_wtd"]  # low => no minutes-weighted blocker => crowding low

    # ---- 2x2 decision screen ----
    tv_med = d["travel_score"].median()
    ft_med = d["fit_score"].median()
    d["travel_hi"] = d["travel_score"] > tv_med
    d["fit_hi"] = d["fit_score"] > ft_med
    d["quadrant"] = np.where(d["travel_hi"] & d["fit_hi"], "additive",
                    np.where(d["travel_hi"] & ~d["fit_hi"], "redundant",
                    np.where(~d["travel_hi"] & d["fit_hi"], "missed-need", "irrelevant")))

    base = float((d["delta"] > 0).mean())
    add = d[d["quadrant"] == "additive"]
    add_rate = float((add["delta"] > 0).mean())

    print("=" * 64)
    print(f"N movers = {len(d)}   base rate P(delta>0) = {base:.3f}")
    print("-" * 64)
    print("TRAVEL axis (origin-only forecast, leave-one-out):")
    print(f"  features           = {feats}")
    print(f"  beta (z-scaled)    = {np.round(beta,3)}")
    print(f"  OOS corr(pred, self_sim) = {r_oos:+.3f}")
    print(f"  OOS accuracy (travelled) = {acc:.3f}   AUC = {auc:.3f}")
    print("-" * 64)
    print("FIT axis (destination hole, measured pre-move = -decoy_sim)")
    print("-" * 64)
    print("2x2 quadrant counts:")
    print(d["quadrant"].value_counts().to_string())
    print(f"\n  ADDITIVE cell (travel_hi & fit_hi): P(delta>0) = {add_rate:.3f} "
          f"(n={len(add)})  vs base {base:.3f}")
    for nm, move in [
        ("M. Bridges", "BKN 2023-24 → NYK 2024-25"),
        ("A. Simons", "POR 2024-25 → BOS 2025-26"),
        ("J. Robinson-Earl", "NOP 2024-25 → IND 2025-26"),
        ("J. Okogie", "PHX 2024-25 → CHA 2024-25"),
    ]:
        r = d[(d["player"] == nm) & (d["move"] == move)]
        if not r.empty:
            r = r.iloc[0]
            print(f"  {r['player']:16s} travel={r['travel_score']:+.2f} "
                  f"fit={r['fit_score']:+.2f}  -> {r['quadrant']}  (delta={r['delta']:+.2f})")

    d.to_csv(OUT_DIR / "travel_fit_screen.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'travel_fit_screen.csv'}")


if __name__ == "__main__":
    main()
