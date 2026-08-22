"""
PORTABILITY STUDY — FULL CONFIRMATORY RUN  §7.2
==============================================
Run after the gate passes (portability_gate_result.txt). Result is recorded in
PORTABILITY_PREREGISTRATION.md §8.

T1 (portability) confirmed iff ALL THREE:
  (1) median_p Δ(p) > 0
  (2) matched-set (within-pool) exchangeability permutation p < 0.01
      (each mover's {self} U {decoys} pool is relabeled; cross-mover / franchise
       dependence is handled by the destination-franchise bootstrap in (3), not here)
  (3) destination-franchise cluster-bootstrap 95% CI of median Δ excludes 0

Δ(p) = cos(ΔA_p^f1, ΔA_p^f2) − mean_{q in D(p)} cos(ΔA_p^f1, ΔA_q^f2)

Run: python portability_full_study.py
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

from portability_common import build_signature_cache, cossim, SIG_COLS, OUT_DIR
from portability_gate import build_movers, PREREG_SEASONS

SEED = 17
N_PERM = 5000
N_BOOT = 2000
CONF_P = 0.01


def compute_deltas(sig, max_gap=0):
    movers, sig_mat, _ = build_movers(sig, max_gap=max_gap)
    recs = []
    for m in movers:
        if m["n_decoys"] < 3:
            continue
        a = sig_mat[m["i1"]]                       # origin anchor ΔA
        self_sim = cossim(a, sig_mat[m["i2"]])     # own destination ΔA
        dsims = [cossim(a, sig_mat[j]) for j in m["decoys"]]
        delta = self_sim - float(np.mean(dsims))
        recs.append(dict(mover_id=m["mover_id"], player_id=m["player_id"],
                         player=m["player"], f1=m["f1"], f2=m["f2"],
                         yy1=m["yy1"], yy2=m["yy2"], gap_seasons=m["gap_seasons"],
                         dest_fr=m["f2"], self_sim=self_sim,
                         decoy_sim=float(np.mean(dsims)), delta=delta,
                         # store the per-mover pool for the permutation null
                         pool=[self_sim] + dsims))
    return pd.DataFrame(recs)


def perm_p(df, n_perm, rng):
    """Exchangeability null: in each mover pool {self} U {decoys}, designate a random
    element as pseudo-self; Δ_null = pseudo_self - mean(rest). Median over movers."""
    obs = float(np.median(df["delta"]))
    pools = [np.asarray(p, float) for p in df["pool"]]
    sums = np.array([p.sum() for p in pools])
    sizes = np.array([len(p) for p in pools])
    cnt = 0
    for _ in range(n_perm):
        d = np.empty(len(pools))
        for i, p in enumerate(pools):
            k = rng.integers(sizes[i])
            d[i] = p[k] - (sums[i] - p[k]) / (sizes[i] - 1)
        if np.median(d) >= obs:
            cnt += 1
    return obs, cnt / n_perm


def boot_ci(df, cluster_col, n_boot, rng):
    groups = {g: sub.index.to_numpy() for g, sub in df.groupby(cluster_col)}
    keys = list(groups)
    meds = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in pick])
        meds.append(float(np.median(df.loc[idx, "delta"])))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gap", type=int, default=0,
                    help="max tolerated missing-season gap between origin and "
                         "destination stints (0=strict adjacency, confirmatory default)")
    args = ap.parse_args()

    t0 = time.time()
    sig = build_signature_cache().reset_index(drop=True)
    sig = sig[sig["yy"].isin(PREREG_SEASONS)].reset_index(drop=True)
    print(f"restricted to pre-registered seasons {PREREG_SEASONS}: "
          f"{len(sig)} player-signatures, {sig['key'].nunique()} team-seasons")
    df = compute_deltas(sig, max_gap=args.max_gap)
    n = len(df)
    if args.max_gap > 0:
        print(f"[robustness run: max_gap={args.max_gap}, NOT the confirmatory default]")
        print(f"  gap_seasons distribution:\n{df['gap_seasons'].value_counts().sort_index().to_string()}")
    n_dest = df["dest_fr"].nunique()
    under = n_dest < 20
    print(f"\nmovers analyzed: {n}   distinct destination franchises: {n_dest}"
          + ("  [UNDERPOWERED <20]" if under else ""))
    print(f"  median self_sim ={np.median(df['self_sim']):+.3f}   "
          f"median decoy_sim ={np.median(df['decoy_sim']):+.3f}")

    obs, p = perm_p(df, N_PERM, np.random.default_rng(SEED + 1))
    # descriptive sign test on the decision rule Δ>0 vs the 50% exchangeability reference
    n_pos = int((df["delta"] > 0).sum())
    try:
        from scipy.stats import binomtest
        sign_p = float(binomtest(n_pos, n, 0.5).pvalue)
    except Exception:
        sign_p = float("nan")
    print(f"   decision rule Δ>0: {n_pos}/{n} = {n_pos/n:.3f}  (sign test vs 0.5 p={sign_p:.2e})")
    lo, hi = boot_ci(df, "dest_fr", N_BOOT, np.random.default_rng(SEED + 2))
    # robustness: origin-franchise and unclustered player bootstrap
    lo_o, hi_o = boot_ci(df, "f1", N_BOOT, np.random.default_rng(SEED + 3))
    df["_solo"] = np.arange(len(df))
    lo_p, hi_p = boot_ci(df, "_solo", N_BOOT, np.random.default_rng(SEED + 4))

    c1 = obs > 0
    c2 = p < CONF_P
    c3 = lo > 0
    t1 = c1 and c2 and c3
    print(f"\nT1 median Δ = {obs:+.4f}")
    print(f"   (1) median>0            : {'PASS' if c1 else 'FAIL'}")
    print(f"   (2) perm p={p:.4f} <0.01 : {'PASS' if c2 else 'FAIL'}")
    print(f"   (3) dest-fr boot CI [{lo:+.4f},{hi:+.4f}] excl 0 : {'PASS' if c3 else 'FAIL'}")
    print(f"       robustness origin-fr CI [{lo_o:+.4f},{hi_o:+.4f}]  "
          f"player CI [{lo_p:+.4f},{hi_p:+.4f}]")

    if t1:
        verdict = ("T1 CONFIRMED — structural role is a PORTABLE PLAYER TRAIT: a player's "
                   "wiring-displacement signature travels across franchises beyond his play-type "
                   "profile. Scoutable.")
    else:
        verdict = ("T1 NOT confirmed — HONEST NEGATIVE: structural role is context-emergent, not a "
                   "portable player trait (explains the V2 absence-simulation failure). Terminal "
                   "test of this line per §9.")
    if under:
        verdict += "  [UNDERPOWERED: <20 destination franchises]"
    print("\n" + "=" * 70)
    print("PORTABILITY CONFIRMATORY RESULT")
    print("=" * 70)
    print(f"  {verdict}")
    print("=" * 70)

    suffix = "" if args.max_gap == 0 else f"_gap{args.max_gap}"
    df.drop(columns=["pool", "_solo"]).to_csv(OUT_DIR / f"portability_deltas{suffix}.csv", index=False)
    with open(OUT_DIR / f"portability_full_result{suffix}.txt", "w", encoding="utf-8") as f:
        f.write("PORTABILITY CONFIRMATORY RESULT\n")
        f.write(f"seed={SEED} n_perm={N_PERM} n_boot={N_BOOT} max_gap={args.max_gap}\n")
        f.write(f"movers={n} dest_franchises={n_dest} under={under}\n\n")
        f.write(f"median self_sim={np.median(df['self_sim']):+.3f} "
                f"decoy_sim={np.median(df['decoy_sim']):+.3f}\n")
        f.write(f"T1 median Δ={obs:+.4f} perm_p={p:.4f}\n")
        f.write(f"decision rule Δ>0: {n_pos}/{n}={n_pos/n:.4f} sign_test_p={sign_p:.3e}\n")
        f.write(f"  dest-fr CI [{lo:+.4f},{hi:+.4f}] (primary)\n")
        f.write(f"  origin-fr CI [{lo_o:+.4f},{hi_o:+.4f}]  player CI [{lo_p:+.4f},{hi_p:+.4f}]\n")
        f.write(f"  (1)median>0={c1} (2)perm<0.01={c2} (3)CI excl0={c3}\n\n")
        f.write(f"VERDICT: {verdict}\n")
    print(f"\nWrote {OUT_DIR / f'portability_full_result{suffix}.txt'} and portability_deltas{suffix}.csv")
    print(f"Total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
