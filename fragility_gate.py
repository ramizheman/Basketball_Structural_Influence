"""
FRAGILITY STUDY — FEASIBILITY GATE
==================================

Pre-registered kill test for FRAGILITY_PREREGISTRATION.md (§7.1). Runs the frozen
8-team pilot and reports the three gate criteria. Any single failure => do not
run the full study.

Structural influence (leave-one-player-out on the Structural Influence association object A):
    L(p) = 1 - cos(A_full, A_{-p})          (primary; §4)
    frobenius reported as labeled robustness only.

Gate criteria (§7.1) — passes only if all three hold:
  G1 well-behaved : for each pilot team-season, every included player's L is finite
                    and > 0, and Spearman(L, usage) > 0.
  G2 decoupling   : pooled R²(L ~ [u, u²]) < 0.90  (usage not a sufficient statistic).
  G3 replicates   : usage-residual r(p) is split-half stable — Spearman(r_odd, r_even)
                    > 0 with permutation p < 0.05.

Reuses the VALIDATED invariant/null/distances from wiring_gate (no drift).

Run:  python fragility_gate.py
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
    PILOT, PLAY_TYPES_8, OUT_DIR,
    pull_edges, team_season_rows, split_halves,
    inclusion_ok, assoc_O, null_O_stats, standardize,
    cos_offdiag, frob_offdiag,
)

# player inclusion (§6, registered)
MIN_INIT_SEASON = 50
MIN_INIT_HALF = 20
# standardization null draws for the GATE (nuisance scale param, not a confirmatory
# setting; the confirmatory §7.2 run uses the registered count).
N_NULL_GATE = 500
N_PERM = 5000
SEED = 17


def build_B_labeled(rows: pd.DataFrame):
    piv = (rows.groupby(["player", "ptype"])["n"].sum()
               .unstack(fill_value=0)
               .reindex(columns=PLAY_TYPES_8, fill_value=0))
    B = piv.to_numpy(dtype=np.int64)
    mask = B.sum(axis=1) > 0
    return B[mask], list(np.asarray(piv.index)[mask])


def loo_loads(B, players, rng, n_null):
    """Leave-one-player-out structural influences on A. Returns dict player -> stats."""
    E, sd, _ = null_O_stats(B, rng, n_null)
    A_full = standardize(assoc_O(B), E, sd)
    total = int(B.sum())
    out = {}
    for idx, p in enumerate(players):
        init = int(B[idx].sum())
        Bm = np.delete(B, idx, axis=0)
        if Bm.shape[0] < 2 or Bm.sum() == 0:
            continue
        Em, sdm, _ = null_O_stats(Bm, rng, n_null)
        Am = standardize(assoc_O(Bm), Em, sdm)
        out[p] = dict(u=init / total, init=init,
                      Lcos=cos_offdiag(A_full, Am),
                      Lfrob=frob_offdiag(A_full, Am))
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return 0.0
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def resid_on_usage(u, L):
    """residual of L after OLS on [1, u, u^2]."""
    u = np.asarray(u, float)
    X = np.column_stack([np.ones_like(u), u, u ** 2])
    y = np.asarray(L, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return y - yhat, r2


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = pull_edges()

    full_rows = []      # (team, franchise, player, u, Lcos, Lfrob, init)
    half_rows = []      # (team, player, half, u, Lcos, init)
    g1_ok = True
    print("\n--- per team-season: leave-one-out influences ---")
    for tri, yy, label, franchise, role in PILOT:
        rows = team_season_rows(df, tri, yy)
        odd, even = split_halves(rows)
        B_full, pl_full = build_B_labeled(rows)
        B_odd, pl_odd = build_B_labeled(odd)
        B_even, pl_even = build_B_labeled(even)
        if not (inclusion_ok(B_odd) and inclusion_ok(B_even)):
            print(f"  {label:<32} EXCLUDED (team-half inclusion)")
            continue

        loads = loo_loads(B_full, pl_full, rng, N_NULL_GATE)
        team_u, team_L = [], []
        for p, s in loads.items():
            if s["init"] >= MIN_INIT_SEASON:
                full_rows.append(dict(team=label, franchise=franchise, player=p,
                                      u=s["u"], Lcos=s["Lcos"], Lfrob=s["Lfrob"], init=s["init"]))
                team_u.append(s["u"])
                team_L.append(s["Lcos"])
        # G1 sanity for this team
        finite_ok = all(np.isfinite(v) and v > 0 for v in team_L) and len(team_L) >= 3
        sp = spearman(team_L, team_u)
        if not (finite_ok and sp > 0):
            g1_ok = False
        print(f"  {label:<32} players={len(team_L):>2}  Spearman(L,u)={sp:+.2f}  "
              f"{'ok' if finite_ok and sp > 0 else 'FLAG'}")

        for B_h, pl_h, htag in [(B_odd, pl_odd, "odd"), (B_even, pl_even, "even")]:
            lh = loo_loads(B_h, pl_h, rng, N_NULL_GATE)
            for p, s in lh.items():
                if s["init"] >= MIN_INIT_HALF:
                    half_rows.append(dict(team=label, player=p, half=htag,
                                          u=s["u"], Lcos=s["Lcos"]))

    full = pd.DataFrame(full_rows)
    half = pd.DataFrame(half_rows)

    # ---- G1 -----------------------------------------------------------------
    print(f"\nG1 well-behaved (L finite>0 & Spearman(L,u)>0 per team): "
          f"{'PASS' if g1_ok else 'FAIL'}")

    # ---- G2 -----------------------------------------------------------------
    _, r2_full = resid_on_usage(full["u"].to_numpy(), full["Lcos"].to_numpy())
    g2 = r2_full < 0.90
    print(f"G2 decoupling (pooled R²(L~[u,u²]) < 0.90): R²={r2_full:.3f}  "
          f"-> {'PASS' if g2 else 'FAIL'}")

    # ---- G3 -----------------------------------------------------------------
    # residual on usage pooled over all (team,player,half); then pair odd vs even
    resid_h, _ = resid_on_usage(half["u"].to_numpy(), half["Lcos"].to_numpy())
    half = half.assign(resid=resid_h)
    piv = half.pivot_table(index=["team", "player"], columns="half",
                           values="resid", aggfunc="first")
    piv = piv.dropna(subset=["odd", "even"])
    r_odd = piv["odd"].to_numpy()
    r_even = piv["even"].to_numpy()
    sp_obs = spearman(r_odd, r_even)
    prng = np.random.default_rng(SEED + 99)
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        perm[i] = spearman(r_odd, prng.permutation(r_even))
    p3 = float((perm >= sp_obs).mean())
    g3 = (sp_obs > 0) and (p3 < 0.05)
    print(f"G3 residual replicates (Spearman(r_odd,r_even)>0, perm p<0.05): "
          f"rho={sp_obs:+.3f}  p={p3:.4f}  n_players={len(piv)}  "
          f"-> {'PASS' if g3 else 'FAIL'}")

    # ---- descriptive: hidden hubs / redundant stars -------------------------
    resid_full, _ = resid_on_usage(full["u"].to_numpy(), full["Lcos"].to_numpy())
    full = full.assign(resid=resid_full)
    print("\n--- descriptive (NOT confirmatory) ---")
    print("  hidden hubs (load >> usage, top residuals):")
    for r in full.nlargest(6, "resid").itertuples(index=False):
        print(f"    {r.player:<22} {r.team[:18]:<18} u={r.u:.3f} L={r.Lcos:.3f} resid={r.resid:+.3f}")
    print("  redundant stars (usage >> load, bottom residuals, u>0.10):")
    hi_u = full[full["u"] > 0.10]
    for r in hi_u.nsmallest(6, "resid").itertuples(index=False):
        print(f"    {r.player:<22} {r.team[:18]:<18} u={r.u:.3f} L={r.Lcos:.3f} resid={r.resid:+.3f}")

    # ---- verdict ------------------------------------------------------------
    gate_pass = g1_ok and g2 and g3
    print("\n" + "=" * 66)
    print("FRAGILITY FEASIBILITY GATE — VERDICT")
    print("=" * 66)
    print(f"  G1 well-behaved : {'PASS' if g1_ok else 'FAIL'}")
    print(f"  G2 decoupling   : {'PASS' if g2 else 'FAIL'}  (R²={r2_full:.3f})")
    print(f"  G3 replicates   : {'PASS' if g3 else 'FAIL'}  (rho={sp_obs:+.3f}, p={p3:.4f})")
    verdict = ("GATE PASSES — fragility is real & usage-decoupled at pilot scale; proceed"
               if gate_pass else
               "GATE FAILS — do NOT run the full fragility study; report the negative")
    print(f"\n  {verdict}")
    print("=" * 66)

    out = OUT_DIR / "fragility_gate_result.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("FRAGILITY FEASIBILITY GATE\n")
        f.write(f"seed={SEED} n_null_gate={N_NULL_GATE} n_perm={N_PERM}\n\n")
        f.write(f"G1 well-behaved : {'PASS' if g1_ok else 'FAIL'}\n")
        f.write(f"G2 decoupling   : {'PASS' if g2 else 'FAIL'} R2={r2_full:.3f}\n")
        f.write(f"G3 replicates   : {'PASS' if g3 else 'FAIL'} rho={sp_obs:+.3f} "
                f"p={p3:.4f} n_players={len(piv)}\n")
        f.write(f"\nVERDICT: {verdict}\n")
    full.to_csv(OUT_DIR / "fragility_gate_loads.csv", index=False)
    print(f"\nWrote {out}")
    print(f"Total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
