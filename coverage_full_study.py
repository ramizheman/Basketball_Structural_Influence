"""
CONFIRMATORY — Structural Role Coverage vs Talent Accumulation  (one-shot)
==========================================================================
Registered spec: TOPOLOGY_COVERAGE_PREREGISTRATION.md, sections 2/5/7.2/9.
Run ONLY after coverage_gate.py PASSED (it did; see pre-reg 8.1).

Primary model M1 (no play-type FE):
  PPP_i = b*COVERED + a1*n_ORG + a2*n_CONN + a3*n_TERM + g*TALENT
          + FE(team-season, absorbed) + dummies(opp, period, score-bucket) + e

COVERED = (>=1 ORGANIZER & >=1 CONNECTOR & >=1 TERMINAL). Because the linear role
counts are in the model and COVERED is their non-linear threshold, b is the
COMPLETENESS effect OVER AND ABOVE accumulation of roles and OVER AND ABOVE talent.

T1 confirmed iff (1) b>0, (2) team-season cluster-robust p<0.01, (3) team-season
cluster-bootstrap 95% CI of b excludes 0.

Robustness (reported, non-decisive): M2 with play-type FE; franchise-clustered
bootstrap; talent control removed (naive b); >=4/5-classified inclusion.

*** ASSOCIATION conditional on measured talent — NOT a causal / talent-independent
optimum (see pre-reg section 4 ceiling). ***

Run: python coverage_full_study.py
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
from role_composition_playmix import pull_possessions, build_class_map
from role_composition_ppp import dummies, demean_by_group, cluster_se, score_bucket
from coverage_common import add_covered, build_talent, SEED, N_BOOT, CONF_P

KEY = ["COVERED", "n_ORGANIZER", "n_CONNECTOR", "n_TERMINAL", "TALENT"]


def two_sided_p(t):
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))


def build_design(df, use_playtype, use_talent=True):
    df = df.copy()
    df["ts"] = df["team"] + "_" + df["yy"].astype(str)
    df["sb"] = score_bucket(df["score_margin"].fillna(0).to_numpy()).astype(str)
    df["per"] = df["period"].fillna(0).astype(int).astype(str)
    ctrl = ["opp", "per", "sb"] + (["ptype"] if use_playtype else [])
    D = dummies(df, ctrl)
    keycols = ["COVERED", "n_ORGANIZER", "n_CONNECTOR", "n_TERMINAL"] + (["TALENT"] if use_talent else [])
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
    """Per team-season ZtZ, Zty (exact building blocks for cluster bootstrap)."""
    blocks = {}
    ser = pd.Series(np.arange(len(ts)))
    for key, idx in ser.groupby(pd.Series(ts)).indices.items():
        idx = np.asarray(idx)
        Zc = Zb[idx]
        blocks[key] = (Zc.T @ Zc, Zc.T @ yb[idx], team[idx][0])
    return blocks


def bootstrap_ci(blocks, cov_idx, group="ts", n_boot=N_BOOT, seed=SEED):
    # collapse blocks to bootstrap-cluster level (ts itself, or franchise=team)
    if group == "ts":
        keys = list(blocks.keys())
        agg = {k: blocks[k][:2] for k in keys}
    else:  # franchise
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
    return float(lo), float(hi), float(est.mean()), float(est.std(ddof=1))


def run_model(df, tag, use_playtype, use_talent, do_boot=("ts",)):
    Zb, yb, zn, ts, team = build_design(df, use_playtype, use_talent)
    beta, se = fit(Zb, yb, ts)
    cov_idx = zn.index("COVERED")
    b = beta[cov_idx]; s = se[cov_idx]; t = b / s; pval = two_sided_p(t)
    print(f"\n=== {tag} ===   (COVERED = all-three-functions present; PPP units, x100=pts/100)")
    print(f"    n={len(df):,}  team-seasons={len(set(ts))}")
    print(f"    {'term':<15}{'beta':>10}{'pts/100':>9}{'clustSE':>10}{'t':>7}")
    for n in KEY if use_talent else [k for k in KEY if k != "TALENT"]:
        i = zn.index(n)
        print(f"    {n:<15}{beta[i]:>10.4f}{beta[i]*100:>9.2f}{se[i]:>10.4f}{beta[i]/se[i]:>7.2f}")
    print(f"    COVERED coef: b={b:.4f} ({b*100:+.2f} pts/100)  clustSE={s:.4f}  t={t:.2f}  p={pval:.2e}")
    # minimum detectable effect at 80% power, two-sided alpha=0.05, given cluster-robust SE
    mde80 = (1.959964 + 0.841621) * s
    print(f"    precision: cluster-robust SE={s*100:.2f} pts/100  MDE(80%%)={mde80*100:.2f} pts/100")
    out = dict(model=tag, beta=b, pts_per_100=b*100, cluster_se=s, t=t, p=pval, n=len(df),
               mde80_pts100=mde80 * 100)
    for n in (KEY if use_talent else [k for k in KEY if k != "TALENT"]):
        i = zn.index(n)
        out[f"coef_{n}_pts100"] = beta[i] * 100
        out[f"t_{n}"] = beta[i] / se[i]
    blocks = None
    for grp in do_boot:
        if blocks is None:
            blocks = cluster_blocks(Zb, yb, ts, team)
        lo, hi, bm, bsd = bootstrap_ci(blocks, cov_idx, group=grp)
        excl = (lo > 0) or (hi < 0)
        print(f"    [{grp}] cluster-bootstrap 95% CI of COVERED: "
              f"[{lo*100:+.2f}, {hi*100:+.2f}] pts/100  excludes 0: {excl}")
        out[f"ci_lo_{grp}"] = lo; out[f"ci_hi_{grp}"] = hi; out[f"ci_excl0_{grp}"] = excl
    return out


def main():
    t0 = time.time()
    df = pull_possessions()
    cmap, u_med, l_med = build_class_map()
    print(f"classifying {len(df):,} possessions ...", flush=True)
    df = add_covered(df, cmap)
    df = df[df["pts"].notna()].copy()

    prim = df[df["n_classified"] == 5].copy()
    print(f"frozen inclusion (all 5 classified): {len(prim):,} possessions; "
          f"overall PPP={prim['pts'].mean():.4f}")
    print("building frozen talent control ...", flush=True)
    prim["TALENT"] = build_talent(prim)

    # ---- PRIMARY: M1, team-season + franchise bootstrap ------------------------
    r_m1 = run_model(prim, "M1 PRIMARY: coverage->PPP (no play-type FE)", False, True,
                     do_boot=("ts", "franchise"))

    # ---- robustness ------------------------------------------------------------
    r_m2 = run_model(prim, "M2 robustness: WITH play-type FE (within-play)", True, True, do_boot=("ts",))
    r_naive = run_model(prim, "NAIVE robustness: talent control REMOVED", False, False, do_boot=("ts",))

    df4 = df[df["n_classified"] >= 4].copy()
    df4["TALENT"] = build_talent(df4)
    r_m4 = run_model(df4, "M1 robustness: >=4/5 classified inclusion", False, True, do_boot=("ts",))

    # ---- verdict (M1 primary) --------------------------------------------------
    confirmed = (r_m1["beta"] > 0) and (r_m1["p"] < CONF_P) and r_m1["ci_excl0_ts"]
    print("\n" + "=" * 70)
    print(f"T1 {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}  "
          f"(b>0: {r_m1['beta']>0}, p<{CONF_P}: {r_m1['p']<CONF_P}, "
          f"team-season CI excludes 0: {r_m1['ci_excl0_ts']})")
    if confirmed:
        print("-> Coverage (creation+connection+conversion present) adds offense OVER AND ABOVE")
        print("   role accumulation and measured talent. Association consistent with a COVERAGE")
        print("   effect, conditional on measured talent (NOT a talent-independent optimum).")
    else:
        print("-> Honest negative: ACCUMULATION, not coverage. Given role counts + talent, having")
        print("   all three functions present adds nothing. Terminal test of the role program.")
    print("=" * 70)

    pd.DataFrame([r_m1, r_m2, r_naive, r_m4]).to_csv(OUT_DIR / "coverage_full_study.csv", index=False)
    print(f"\nWrote {OUT_DIR/'coverage_full_study.csv'}")
    print("REMINDER: association conditional on measured talent; talent control under-measures")
    print("connectors by construction (pre-reg section 4 ceiling).")
    print(f"Total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
