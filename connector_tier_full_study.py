"""
CONFIRMATORY-DISCIPLINE — Hub / Marginal-Big / Marginal-Wing tier split vs Talent
==================================================================================
Same exact machinery as coverage_full_study.py / hasconn_full_study.py, applied to
the three-way split of the Connector/Hub class (Eq. 6 taxonomy, split at the
within-class median residual load, marginal half further split by position).

Primary model M1 (no play-type FE):
  PPP_i = b1*HasHub + b2*HasMargBig + b3*HasMargWing
          + a1*n_ORG + a2*n_TERM + a3*n_HUBCONN + a4*n_MARGBIG + a5*n_MARGWING
          + g*TALENT + FE(team-season) + dummies(opp, period, score-bucket) + e

Run: python connector_tier_full_study.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from math import erf, sqrt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from wiring_gate import OUT_DIR
from role_composition_playmix import pull_possessions
from role_composition_ppp import dummies, demean_by_group, cluster_se, score_bucket
from coverage_common import build_talent, SEED, N_BOOT, CONF_P
from _connector_tier_ppp_3way import build_3way_class_map, add_composition6

TERMS = ["HasHub", "HasMargBig", "HasMargWing"]
# Presence indicators only (no own-count collinear controls); n_ORG / n_TERM retained.
KEY = TERMS + ["n_ORGANIZER", "n_TERMINAL", "TALENT"]


def two_sided_p(t):
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))


def build_design(df, use_playtype, use_talent=True):
    df = df.copy()
    df["ts"] = df["team"] + "_" + df["yy"].astype(str)
    df["sb"] = score_bucket(df["score_margin"].fillna(0).to_numpy()).astype(str)
    df["per"] = df["period"].fillna(0).astype(int).astype(str)
    ctrl = ["opp", "per", "sb"] + (["ptype"] if use_playtype else [])
    D = dummies(df, ctrl)
    keycols = TERMS + ["n_ORGANIZER", "n_TERMINAL"] + (["TALENT"] if use_talent else [])
    Xkey = df[keycols].astype(float)
    Z = pd.concat([Xkey, D], axis=1)
    znames = list(Z.columns)
    y = df["pts"].to_numpy(float)
    ts = df["ts"].to_numpy()
    team = df["team"].to_numpy()
    yb = pd.Series(y); yb = (yb - yb.groupby(pd.Series(ts)).transform("mean")).to_numpy()
    Zb = demean_by_group(Z.to_numpy(float), ts)
    return Zb, yb, znames, ts, team


def fit(Zb, yb, ts):
    beta, *_ = np.linalg.lstsq(Zb, yb, rcond=None)
    resid = yb - Zb @ beta
    se = cluster_se(Zb, resid, beta, ts)
    return beta, se


def cluster_blocks(Zb, yb, ts, team):
    blocks = {}
    ser = pd.Series(np.arange(len(ts)))
    for key, idx in ser.groupby(pd.Series(ts)).indices.items():
        idx = np.asarray(idx)
        Zc = Zb[idx]
        blocks[key] = (Zc.T @ Zc, Zc.T @ yb[idx], team[idx][0])
    return blocks


def bootstrap_ci(blocks, cov_idx, group="ts", n_boot=N_BOOT, seed=SEED):
    if group == "ts":
        keys = list(blocks.keys())
        agg = {k: blocks[k][:2] for k in keys}
    else:
        agg = {}
        for k, (ZtZ, Zty, tm) in blocks.items():
            if tm not in agg:
                agg[tm] = [ZtZ.copy(), Zty.copy()]
            else:
                agg[tm][0] += ZtZ; agg[tm][1] += Zty
        keys = list(agg.keys())
        agg = {k: (v[0], v[1]) for k, v in agg.items()}
    p = len(next(iter(agg.values()))[0])
    rng = np.random.default_rng(seed)
    nk = len(keys)
    karr = np.array(keys, dtype=object)
    est = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, nk, nk)
        A = np.zeros((p, p)); c = np.zeros(p)
        for j in pick:
            ZtZ, Zty = agg[karr[j]]
            A += ZtZ; c += Zty
        try:
            beta = np.linalg.solve(A, c)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, c, rcond=None)[0]
        est[b] = beta[cov_idx]
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(lo), float(hi)


def run_model(df, tag, use_playtype, use_talent, do_boot=("ts",)):
    Zb, yb, zn, ts, team = build_design(df, use_playtype, use_talent)
    beta, se = fit(Zb, yb, ts)
    print(f"\n=== {tag} ===")
    print(f"    n={len(df):,}  team-seasons={len(set(ts))}")
    print(f"    {'term':<15}{'pts/100':>9}{'clustSE':>10}{'t':>7}{'p':>10}")
    out = dict(model=tag, n=len(df))
    blocks = None
    for term in KEY if use_talent else [k for k in KEY if k != "TALENT"]:
        i = zn.index(term)
        b, s = beta[i], se[i]
        t = b / s if s > 0 else np.nan
        p = two_sided_p(t)
        print(f"    {term:<15}{b*100:>9.2f}{s:>10.4f}{t:>7.2f}{p:>10.2e}")
        out[f"coef_{term}_pts100"] = b * 100
        out[f"se_{term}"] = s
        out[f"t_{term}"] = t
        out[f"p_{term}"] = p
        if term in TERMS:
            for grp in do_boot:
                if blocks is None:
                    blocks = cluster_blocks(Zb, yb, ts, team)
                lo, hi = bootstrap_ci(blocks, i, group=grp)
                out[f"ci_lo_{grp}_{term}"] = lo * 100
                out[f"ci_hi_{grp}_{term}"] = hi * 100
                print(f"      [{grp}] boot 95% CI: [{lo*100:+.2f}, {hi*100:+.2f}] pts/100")
    return out


def main():
    t0 = time.time()
    df = pull_possessions()
    cmap, u_med, r_med = build_3way_class_map()
    print(f"classifying {len(df):,} possessions ...", flush=True)
    df = add_composition6(df, cmap)
    df["HasHub"] = (df["n_HUBCONN"] >= 1).astype(int)
    df["HasMargBig"] = (df["n_MARGBIG"] >= 1).astype(int)
    df["HasMargWing"] = (df["n_MARGWING"] >= 1).astype(int)
    df = df[df["pts"].notna()].copy()

    prim = df[df["n_classified"] == 5].copy()
    print(f"frozen inclusion (all 5 classified): {len(prim):,} possessions; "
          f"overall PPP={prim['pts'].mean():.4f}")
    print("building frozen talent control ...", flush=True)
    prim["TALENT"] = build_talent(prim)

    r_m1 = run_model(prim, "M1 PRIMARY: 3-way tier split -> PPP (TALENT-controlled)", False, True,
                     do_boot=("ts", "franchise"))
    r_naive = run_model(prim, "NAIVE robustness: talent control REMOVED", False, False, do_boot=("ts",))
    r_m2 = run_model(prim, "M2 robustness: WITH play-type FE (within-play)", True, True, do_boot=("ts",))

    df4 = df[df["n_classified"] >= 4].copy()
    df4["TALENT"] = build_talent(df4)
    r_m4 = run_model(df4, "M1 robustness: >=4/5 classified inclusion", False, True, do_boot=("ts",))

    print("\n" + "=" * 78)
    print("VERDICT (talent-controlled primary, team-season cluster-bootstrap CI)")
    for term in TERMS:
        conf = (r_m1[f"coef_{term}_pts100"] > 0) and (r_m1[f"p_{term}"] < CONF_P) and \
               (r_m1[f"ci_lo_ts_{term}"] > 0 or r_m1[f"ci_hi_ts_{term}"] < 0)
        print(f"  {term:<14} b={r_m1[f'coef_{term}_pts100']:+.2f} pts/100  p={r_m1[f'p_{term}']:.4f}  "
              f"CI=[{r_m1[f'ci_lo_ts_{term}']:+.2f},{r_m1[f'ci_hi_ts_{term}']:+.2f}]  "
              f"{'CONFIRMED @0.01' if conf else 'not at 0.01 bar'}")
    print("=" * 78)

    pd.DataFrame([r_m1, r_naive, r_m2, r_m4]).to_csv(OUT_DIR / "connector_tier_full_study.csv", index=False)
    print(f"\nWrote {OUT_DIR/'connector_tier_full_study.csv'}")
    print(f"Total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
