"""
CONFIGURATION -> SCORING  (fixed-effects PPP model; DESCRIPTIVE association)
============================================================================
Part (b) of the role-composition analysis. Regress possession points on the
on-court ROLE COMPOSITION, with rich fixed effects, at n~650k possessions.

  PPP_i = b0 + b_conn*n_CONNECTOR + b_org*n_ORGANIZER + b_term*n_TERMINAL
             + b_orgconn*(n_ORG*n_CONN)            [fit / interaction]
             + n_classified
             + FE(team-season)  [absorbed]
             + dummies(opponent, period, score-margin bucket)
             + [play_type dummies in M2 only]

OCCUPANT is the reference class, so each b is the PPP change from swapping one
OCCUPANT for one player of that class, holding the rest fixed.

Two models decompose the mediation the user described (config -> behavior -> points):
  M1  WITHOUT play_type  : total composition->PPP (includes the play-selection path)
  M2  WITH play_type FE  : composition->PPP HOLDING the play fixed (within-play-type
                           efficiency). (M1 - M2) ~ the part routed through changing
                           which plays get run.

SE: cluster-robust by offensive team-season.

*** THIS IS ASSOCIATION, NOT CAUSATION. *** Lineup composition is chosen, not
assigned: a class being off-floor overlaps with bench-unit talent and game state,
which team-season/opponent/period/score controls REDUCE but do not eliminate.
"Adding a connector changes scoring" is NOT licensed; the coefficients describe
conditional associations only.

Run: python role_composition_ppp.py
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

from wiring_gate import PLAY_TYPES_8, OUT_DIR
from role_composition_playmix import pull_possessions, build_class_map, ORDER

KEY = ["n_CONNECTOR", "n_ORGANIZER", "n_TERMINAL"]   # OCCUPANT = reference


def score_bucket(m):
    edges = [-1e9, -15, -8, -3, 3, 8, 15, 1e9]
    return np.digitize(m, edges)


def add_composition(df, cmap):
    plcols = [f"pl{i}" for i in range(5)]
    yy = df["yy"].to_numpy(); team = df["team"].to_numpy()
    comp = {k: np.zeros(len(df), dtype=np.int16) for k in ORDER}
    ncl = np.zeros(len(df), dtype=np.int16)
    for c in plcols:
        names = df[c].to_numpy()
        arr = np.array([cmap.get((team[i], int(yy[i]), names[i]), "?") for i in range(len(df))],
                       dtype=object)
        for k in ORDER:
            comp[k] += (arr == k)
        ncl += (arr != "?")
    for k in ORDER:
        df[f"n_{k}"] = comp[k]
    df["n_classified"] = ncl
    return df


def dummies(df, cols):
    parts = []
    for c in cols:
        d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True, dtype=float)
        parts.append(d)
    return pd.concat(parts, axis=1)


def demean_by_group(mat, groups):
    """Absorb one FE: subtract group means from each column (and return means removed)."""
    out = mat.copy()
    g = pd.Series(groups)
    for j in range(mat.shape[1]):
        s = pd.Series(mat[:, j])
        out[:, j] = (s - s.groupby(g).transform("mean")).to_numpy()
    return out


def cluster_se(X, resid, beta, clusters):
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in pd.Series(np.arange(len(clusters))).groupby(clusters).indices.items():
        idx = np.asarray(idx)
        Xg = X[idx]; ug = resid[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    return np.sqrt(np.diag(V))


def fit_fe(df, use_playtype):
    df = df.copy()
    df["ts"] = df["team"] + "_" + df["yy"].astype(str)
    df["sb"] = score_bucket(df["score_margin"].fillna(0).to_numpy()).astype(str)
    df["per"] = df["period"].fillna(0).astype(int).astype(str)

    ctrl_cols = ["opp", "per", "sb"] + (["ptype"] if use_playtype else [])
    D = dummies(df, ctrl_cols)
    Xkey = df[KEY + ["n_classified"]].astype(float).copy()
    Xkey["org_x_conn"] = df["n_ORGANIZER"].astype(float) * df["n_CONNECTOR"].astype(float)
    Z = pd.concat([Xkey, D], axis=1)
    znames = list(Z.columns)

    y = df["pts"].to_numpy(float)
    Zm = Z.to_numpy(float)
    ts = df["ts"].to_numpy()

    # absorb team-season FE by within-demeaning y and Z
    yb = pd.Series(y); yb = (yb - yb.groupby(pd.Series(ts)).transform("mean")).to_numpy()
    Zb = demean_by_group(Zm, ts)

    beta, *_ = np.linalg.lstsq(Zb, yb, rcond=None)
    resid = yb - Zb @ beta
    se = cluster_se(Zb, resid, beta, ts)
    tstat = beta / se
    return znames, beta, se, tstat, len(df)


def report(tag, znames, beta, se, tstat):
    idx = {n: i for i, n in enumerate(znames)}
    show = KEY + ["org_x_conn", "n_classified"]
    print(f"\n=== {tag}  (per +1 player vs OCCUPANT; PPP units; x100 = pts/100 poss) ===")
    print(f"{'term':<16}{'beta':>9}{'pts/100':>9}{'clust SE':>10}{'t':>7}")
    rows = []
    for n in show:
        i = idx[n]
        print(f"{n:<16}{beta[i]:>9.4f}{beta[i]*100:>9.2f}{se[i]:>10.4f}{tstat[i]:>7.2f}")
        rows.append(dict(model=tag, term=n, beta=beta[i], pts_per_100=beta[i]*100,
                         cluster_se=se[i], t=tstat[i]))
    return rows


def main():
    t0 = time.time()
    df = pull_possessions()
    cmap, _, _ = build_class_map()
    print(f"classifying {len(df):,} possessions for PPP model ...", flush=True)
    df = add_composition(df, cmap)
    df = df[df["n_classified"] >= 4].copy()
    df = df[df["pts"].notna()]
    print(f"possessions used (>=4/5 classified, pts present): {len(df):,}")
    print(f"overall PPP = {df['pts'].mean():.4f}")

    n1, b1, s1, t1, N1 = fit_fe(df, use_playtype=False)
    r1 = report("M1: composition -> PPP (NO play-type FE; total)", n1, b1, s1, t1)
    n2, b2, s2, t2, N2 = fit_fe(df, use_playtype=True)
    r2 = report("M2: composition -> PPP (WITH play-type FE; within-play efficiency)", n2, b2, s2, t2)

    # decomposition note
    idx1 = {n: i for i, n in enumerate(n1)}; idx2 = {n: i for i, n in enumerate(n2)}
    print("\n=== decomposition: total (M1) vs within-play (M2); "
          "difference ~ routed through changing play selection ===")
    print(f"{'term':<16}{'M1 pts/100':>12}{'M2 pts/100':>12}{'via play-mix':>14}")
    for n in KEY + ["org_x_conn"]:
        m1v, m2v = b1[idx1[n]]*100, b2[idx2[n]]*100
        print(f"{n:<16}{m1v:>12.2f}{m2v:>12.2f}{m1v-m2v:>14.2f}")

    out = pd.DataFrame(r1 + r2)
    out.to_csv(OUT_DIR / "role_composition_ppp.csv", index=False)
    print(f"\nWrote {OUT_DIR/'role_composition_ppp.csv'}")
    print("REMINDER: association only; lineup composition is chosen, not assigned.")
    print(f"Total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
