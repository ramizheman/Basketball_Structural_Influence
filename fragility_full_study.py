"""
FRAGILITY STUDY — FULL CONFIRMATORY RUN
=======================================

Runs the confirmatory test registered in FRAGILITY_PREREGISTRATION.md (§7.2, as
amended 2026-07-12 / Amendment 1) across ALL team-seasons, not the 8-team pilot.

Only run AFTER the feasibility gate (fragility_gate.py) has passed. It has:
output/fragility_gate_result.txt (G1/G2/G3 all PASS).

Confirmatory claims (§2):
  F1a decoupling : structural influence is NOT reducible to usage.
       confirmed iff pooled R²(L ~ [1,u,u²]) < 0.90 AND the residual share (1-R²)
       has a franchise cluster-bootstrap 95% CI lower bound > 0.10.
  F1b stability  : the usage-residual influence is a reproducible player property.
       CO-PRIMARY (Amendment 1) — BOTH must clear, else F1b NOT confirmed:
         (A) pooled-residual split-half Spearman(r_odd, r_even) > 0
         (B) usage-partial split-half Spearman (control u_odd,u_even) > 0
       each with franchise-clustered permutation p < 0.01 and cluster-bootstrap
       95% CI excluding 0.

Primary metric: L(p) = 1 - cos(A_full, A_{-p})  (§4). Frobenius = labeled robustness.

Non-independence (§6): unit = franchise (= tricode); franchise-clustered permutation
and franchise cluster-bootstrap. Players nest in team-seasons nest in franchises.

Reused with NO drift: invariant/null/distances (wiring_gate), all-teams edge pull +
split (wiring_full_study), leave-one-out influences (fragility_gate).

Run:
    python fragility_full_study.py                 # full run
    python fragility_full_study.py --max-teams 12  # smoke test
    python fragility_full_study.py --n-null 1000 --n-perm 5000
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from wiring_gate import inclusion_ok
from wiring_full_study import pull_edges, split_halves, OUT_DIR
from fragility_gate import (
    loo_loads, spearman,
    MIN_INIT_SEASON, MIN_INIT_HALF,
)
from portability_common import build_B_labeled_by_id

SEED = 17
CONF_P = 0.01           # §7.2
F1A_CI_FLOOR = 0.10     # residual-share lower-bound floor (§7.2)
N_BOOT = 2000


def resid_on_usage(u, L):
    u = np.asarray(u, float)
    X = np.column_stack([np.ones_like(u), u, u ** 2])
    y = np.asarray(L, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return y - yhat, r2


def _rank(a):
    return pd.Series(np.asarray(a, float)).rank().to_numpy()


def _resid_lstsq(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(L1, L2, u1, u2):
    """Partial Spearman corr(L1,L2) controlling for u1,u2 (Amendment 1 test B)."""
    rL1, rL2 = _rank(L1), _rank(L2)
    ru1, ru2 = _rank(u1), _rank(u2)
    X = np.column_stack([np.ones_like(ru1), ru1, ru2])
    e1 = _resid_lstsq(rL1, X)
    e2 = _resid_lstsq(rL2, X)
    d = np.sqrt((e1 ** 2).sum() * (e2 ** 2).sum())
    return float((e1 * e2).sum() / d) if d > 0 else 0.0


# --------------------------------------------------------- build all loads
def build_loads(df, rng, n_null, max_teams=None):
    """Per-player leave-one-out influences for every included team-season.
    Returns (full_df, half_df). full: season-level (init>=50). half: per-half (init>=20)."""
    combos = (df.groupby(["team", "season_yy"])["n"].sum()
                .reset_index().sort_values(["season_yy", "team"]))
    if max_teams is not None:
        combos = combos.head(max_teams)

    full_rows, half_rows = [], []
    for tri, yy, _ in combos.itertuples(index=False):
        rows = df[(df["team"] == tri) & (df["season_yy"] == yy) & (df["is_playoff"] == 0)]
        if rows.empty:
            continue
        odd, even = split_halves(rows)
        # player_id-keyed (NOT display-name-keyed): abbreviated names like
        # "J. Williams" or "S. Curry" collide across real players who were
        # literally on the SAME roster in the SAME season (e.g. Jalen Williams
        # + Jaylin Williams on OKC every season 2021-22..2024-25; Steph + Seth
        # Curry on GSW 2025-26; Jalen + Jeff Green on HOU 2022-23/2023-24).
        # Grouping by name there merges two real players' initiations into one
        # row, corrupting the association matrix and null model for the WHOLE
        # roster, not just the collided players' own loads (confirmed via
        # _tmp_fragility_collision_check.py: SGA's OKC 2023-24 load moved from
        # 0.467 to ~0.28 once OKC's Williams collision is fixed).
        B_odd, ids_odd, id_to_name_odd = build_B_labeled_by_id(odd)
        B_even, ids_even, id_to_name_even = build_B_labeled_by_id(even)
        if not (inclusion_ok(B_odd) and inclusion_ok(B_even)):
            continue
        key = f"{tri}_{yy}"
        B_full, ids_full, id_to_name = build_B_labeled_by_id(rows)

        for pid, s in loo_loads(B_full, ids_full, rng, n_null).items():
            if s["init"] >= MIN_INIT_SEASON:
                full_rows.append(dict(key=key, tri=tri, yy=int(yy),
                                      player_id=pid, player=id_to_name.get(pid, str(pid)),
                                      u=s["u"], Lcos=s["Lcos"], Lfrob=s["Lfrob"]))
        for B_h, ids_h, names_h, htag in [(B_odd, ids_odd, id_to_name_odd, "odd"),
                                           (B_even, ids_even, id_to_name_even, "even")]:
            for pid, s in loo_loads(B_h, ids_h, rng, n_null).items():
                if s["init"] >= MIN_INIT_HALF:
                    half_rows.append(dict(key=key, tri=tri,
                                          player_id=pid, player=names_h.get(pid, str(pid)),
                                          half=htag, u=s["u"], Lcos=s["Lcos"]))
        print(f"  built {key:<10} players_full={sum(1 for r in full_rows if r['key']==key):>2}",
              flush=True)
    return pd.DataFrame(full_rows), pd.DataFrame(half_rows)


# --------------------------------------------------------- F1a
def f1a(full, boot_rng, n_boot, load_col="Lcos"):
    _, r2 = resid_on_usage(full["u"].to_numpy(), full[load_col].to_numpy())
    by_fr = {t: g.index.to_numpy() for t, g in full.groupby("tri")}
    fr = list(by_fr)
    shares = []
    for _ in range(n_boot):
        pick = boot_rng.integers(0, len(fr), size=len(fr))
        idx = np.concatenate([by_fr[fr[i]] for i in pick])
        sub = full.loc[idx]
        _, r2b = resid_on_usage(sub["u"].to_numpy(), sub[load_col].to_numpy())
        shares.append(1.0 - r2b)
    lo, hi = np.percentile(shares, [2.5, 97.5])
    return dict(r2=r2, share=1.0 - r2, ci=(float(lo), float(hi)))


# --------------------------------------------------------- F1b (paired players)
def build_pairs(half):
    """Pooled usage-residual per (player, half), then pair odd vs even per
    (key, player_id). Pivot on player_id so abbreviated names stay distinct.
    """
    resid, _ = resid_on_usage(half["u"].to_numpy(), half["Lcos"].to_numpy())
    half = half.assign(resid=resid)
    piv = half.pivot_table(index=["key", "tri", "player_id"], columns="half",
                           values=["resid", "Lcos", "u"], aggfunc="first").dropna()
    out = pd.DataFrame({
        "tri": piv.index.get_level_values("tri"),
        "r_odd": piv[("resid", "odd")].to_numpy(),
        "r_even": piv[("resid", "even")].to_numpy(),
        "L_odd": piv[("Lcos", "odd")].to_numpy(),
        "L_even": piv[("Lcos", "even")].to_numpy(),
        "u_odd": piv[("u", "odd")].to_numpy(),
        "u_even": piv[("u", "even")].to_numpy(),
    })
    return out.reset_index(drop=True)


def _stat_pooled(pairs):
    return spearman(pairs["r_odd"].to_numpy(), pairs["r_even"].to_numpy())


def _stat_partial(pairs):
    return partial_spearman(pairs["L_odd"], pairs["L_even"], pairs["u_odd"], pairs["u_even"])


def franchise_perm_p(pairs, stat_fn, second_col, n_perm, rng):
    """Franchise-clustered permutation: shuffle `second_col` WITHIN each franchise
    (players exchangeable within franchise under H0), breaking the cross-half player
    link while preserving franchise-level structure (§6)."""
    obs = stat_fn(pairs)
    by_fr = {t: g.index.to_numpy() for t, g in pairs.groupby("tri")}
    base = pairs.copy()
    cnt = 0
    for _ in range(n_perm):
        perm = base.copy()
        col = perm[second_col].to_numpy().copy()
        for idx in by_fr.values():
            col[idx] = rng.permutation(col[idx])
        perm[second_col] = col
        if stat_fn(perm) >= obs:
            cnt += 1
    return obs, cnt / n_perm


def franchise_boot_ci(pairs, stat_fn, n_boot, rng):
    by_fr = {t: g.index.to_numpy() for t, g in pairs.groupby("tri")}
    fr = list(by_fr)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(fr), size=len(fr))
        idx = np.concatenate([by_fr[fr[i]] for i in pick])
        vals.append(stat_fn(pairs.loc[idx].reset_index(drop=True)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=500,
                    help="standardization null draws (nuisance scale param)")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--max-teams", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = pull_edges()

    print("\n--- building leave-one-out influences (all team-seasons, inclusion §6) ---")
    full, half = build_loads(df, rng, args.n_null, max_teams=args.max_teams)
    n_ts = full["key"].nunique()
    n_fr = full["tri"].nunique()
    print(f"\nteam-seasons included : {n_ts}")
    print(f"distinct franchises   : {n_fr}  (effective inferential unit)")
    print(f"season-level players  : {len(full)}")
    if n_fr < 20:
        print("  WARNING (§9): effective N < ~20 franchises — treat as underpowered.")

    boot_rng = np.random.default_rng(SEED + 7)
    perm_rng = np.random.default_rng(SEED + 99)

    # ---- F1a --------------------------------------------------------------
    a_cos = f1a(full, boot_rng, N_BOOT, "Lcos")
    a_frob = f1a(full, np.random.default_rng(SEED + 8), N_BOOT, "Lfrob")  # robustness
    f1a_ok = (a_cos["r2"] < 0.90) and (a_cos["ci"][0] > F1A_CI_FLOOR)
    print("\n--- F1a decoupling ---")
    print(f"  cosine   R²={a_cos['r2']:.3f}  residual-share={a_cos['share']:.3f} "
          f"95%CI[{a_cos['ci'][0]:.3f},{a_cos['ci'][1]:.3f}]  -> {'PASS' if f1a_ok else 'FAIL'}")
    print(f"  frobenius(robustness) R²={a_frob['r2']:.3f} share={a_frob['share']:.3f} "
          f"CI[{a_frob['ci'][0]:.3f},{a_frob['ci'][1]:.3f}]")

    # ---- F1b (co-primary) -------------------------------------------------
    pairs = build_pairs(half)
    print(f"\n--- F1b stability (paired players n={len(pairs)}) ---")
    sp_obs, sp_p = franchise_perm_p(pairs, _stat_pooled, "r_even", args.n_perm, perm_rng)
    sp_lo, sp_hi = franchise_boot_ci(pairs, _stat_pooled,
                                     N_BOOT, np.random.default_rng(SEED + 11))
    pooled_ok = (sp_obs > 0) and (sp_p < CONF_P) and (sp_lo > 0)
    print(f"  (A) pooled-residual  rho={sp_obs:+.3f}  p={sp_p:.4f} "
          f"95%CI[{sp_lo:+.3f},{sp_hi:+.3f}]  -> {'PASS' if pooled_ok else 'FAIL'}")

    # for the partial test, permute L_even within franchise (its cross-half link)
    pa_obs, pa_p = franchise_perm_p(pairs, _stat_partial, "L_even",
                                    args.n_perm, np.random.default_rng(SEED + 123))
    pa_lo, pa_hi = franchise_boot_ci(pairs, _stat_partial,
                                     N_BOOT, np.random.default_rng(SEED + 13))
    partial_ok = (pa_obs > 0) and (pa_p < CONF_P) and (pa_lo > 0)
    print(f"  (B) usage-partial    rho={pa_obs:+.3f}  p={pa_p:.4f} "
          f"95%CI[{pa_lo:+.3f},{pa_hi:+.3f}]  -> {'PASS' if partial_ok else 'FAIL'}")

    f1b_ok = pooled_ok and partial_ok
    f1b_disagree = pooled_ok != partial_ok

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 68)
    print("FRAGILITY CONFIRMATORY RESULT")
    print("=" * 68)
    print(f"  F1a decoupling : {'CONFIRMED' if f1a_ok else 'NOT confirmed'}")
    print(f"  F1b stability  : {'CONFIRMED' if f1b_ok else 'NOT confirmed'}"
          + ("  (co-primary DISAGREEMENT — Amendment 1: NOT confirmed)" if f1b_disagree else ""))
    both = f1a_ok and f1b_ok
    verdict = ("F1 CONFIRMED — offenses have identifiable structural hubs distinct from usage"
               if both else "F1 NOT fully confirmed — see rows above (honest negative per §9)")
    print(f"\n  {verdict}")
    print("=" * 68)

    out = OUT_DIR / "fragility_full_result.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("FRAGILITY CONFIRMATORY RESULT\n")
        f.write(f"seed={SEED} n_null={args.n_null} n_perm={args.n_perm} "
                f"max_teams={args.max_teams}\n")
        f.write(f"team_seasons={n_ts} franchises={n_fr} season_players={len(full)} "
                f"paired_players={len(pairs)}\n\n")
        f.write(f"F1a cosine    R2={a_cos['r2']:.3f} share={a_cos['share']:.3f} "
                f"CI[{a_cos['ci'][0]:.3f},{a_cos['ci'][1]:.3f}]\n")
        f.write(f"F1a frobenius R2={a_frob['r2']:.3f} share={a_frob['share']:.3f} "
                f"CI[{a_frob['ci'][0]:.3f},{a_frob['ci'][1]:.3f}] (robustness)\n")
        f.write(f"F1b (A) pooled  rho={sp_obs:+.3f} p={sp_p:.4f} "
                f"CI[{sp_lo:+.3f},{sp_hi:+.3f}]\n")
        f.write(f"F1b (B) partial rho={pa_obs:+.3f} p={pa_p:.4f} "
                f"CI[{pa_lo:+.3f},{pa_hi:+.3f}]\n\n")
        f.write(f"F1a={'CONFIRMED' if f1a_ok else 'no'} "
                f"F1b={'CONFIRMED' if f1b_ok else 'no'} disagree={f1b_disagree}\n")
        f.write(f"VERDICT: {verdict}\n")
    full.to_csv(OUT_DIR / "fragility_full_loads.csv", index=False)
    print(f"\nWrote {out}")
    print(f"Total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
