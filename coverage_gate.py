"""
FEASIBILITY GATE — Structural Role Coverage vs Talent Accumulation
==================================================================
Registered spec: TOPOLOGY_COVERAGE_PREREGISTRATION.md, section 7.1.

Checks feasibility ONLY. Does NOT compute the coverage effect beta, and does NOT
compute the covered-vs-not PPP contrast (no optional stopping / no peeking).

Passes iff ALL THREE hold:
  G1  coverage varies:      >=20 team-seasons have >=200 possessions in EACH of
                            COVERED and NOT-COVERED.
  G2  separable:            |Spearman(COVERED, TALENT)| < 0.80  AND  the linear role
                            counts do not near-perfectly explain COVERED (aux OLS
                            R^2 < 0.95) -> the threshold carries independent variation.
  G3  talent behaves:       TALENT has positive raw (Spearman) association with PPP.

Any single failure => do NOT run the confirmatory (section 9).

Run: python coverage_gate.py
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

from wiring_gate import OUT_DIR
from role_composition_playmix import pull_possessions, build_class_map
from coverage_common import add_covered, build_talent

MIN_POSS_STRATUM = 200
MIN_TS = 20
MAX_RHO_TALENT = 0.80
MAX_AUX_R2 = 0.95


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    t0 = time.time()
    df = pull_possessions()
    cmap, u_med, l_med = build_class_map()
    print(f"classifying {len(df):,} possessions ...", flush=True)
    df = add_covered(df, cmap)

    # frozen primary inclusion: all 5 on-court classified, pts present
    df = df[(df["n_classified"] == 5) & df["pts"].notna()].copy()
    print(f"role thresholds: usage median={u_med:.3f}  influence median={l_med:.3f}")
    print(f"possessions with all 5 classified + pts present: {len(df):,}")

    print("building frozen talent control (leave-game-out points-per-initiation) ...", flush=True)
    df["TALENT"] = build_talent(df)

    df["ts"] = df["team"] + "_" + df["yy"].astype(str)

    # ---- G1: coverage varies within team-seasons -------------------------------
    g = df.groupby("ts")["COVERED"]
    per_ts = g.agg(cov=lambda s: int((s == 1).sum()), notcov=lambda s: int((s == 0).sum()))
    ok_ts = per_ts[(per_ts["cov"] >= MIN_POSS_STRATUM) & (per_ts["notcov"] >= MIN_POSS_STRATUM)]
    n_ok = len(ok_ts)
    cov_rate = float(df["COVERED"].mean())
    G1 = n_ok >= MIN_TS
    print(f"\nG1 coverage varies within team-seasons")
    print(f"    overall COVERED rate = {cov_rate:.1%}")
    print(f"    team-seasons with >={MIN_POSS_STRATUM} in BOTH covered & not-covered: "
          f"{n_ok} (need >={MIN_TS})  -> {'PASS' if G1 else 'FAIL'}")

    # ---- G2: COVERED separable from talent and from linear counts --------------
    rho_talent = spearman(df["COVERED"].to_numpy(), df["TALENT"].to_numpy())
    # aux OLS: COVERED ~ 1 + n_ORG + n_CONN + n_TERM
    Xa = np.column_stack([np.ones(len(df)),
                          df["n_ORGANIZER"].to_numpy(float),
                          df["n_CONNECTOR"].to_numpy(float),
                          df["n_TERMINAL"].to_numpy(float)])
    ya = df["COVERED"].to_numpy(float)
    ba, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
    resid = ya - Xa @ ba
    aux_r2 = 1.0 - float(resid.var() / ya.var())
    G2 = (abs(rho_talent) < MAX_RHO_TALENT) and (aux_r2 < MAX_AUX_R2)
    print(f"\nG2 coverage separable from talent and from linear counts")
    print(f"    |Spearman(COVERED, TALENT)| = {abs(rho_talent):.3f} (need <{MAX_RHO_TALENT})")
    print(f"    aux R^2 COVERED~linear counts = {aux_r2:.3f} (need <{MAX_AUX_R2})  "
          f"-> {'PASS' if G2 else 'FAIL'}")

    # ---- G3: talent control behaves (positive raw assoc with PPP) --------------
    rho_pps = spearman(df["TALENT"].to_numpy(), df["pts"].to_numpy())
    G3 = rho_pps > 0
    print(f"\nG3 talent control behaves")
    print(f"    Spearman(TALENT, PPP) = {rho_pps:+.3f} (need >0)  -> {'PASS' if G3 else 'FAIL'}")

    # ---- descriptive context (NOT a decision input) ----------------------------
    print(f"\ncontext (not a gate criterion): mean TALENT={df['TALENT'].mean():.3f}, "
          f"distinct team-seasons={df['ts'].nunique()}, distinct opponents={df['opp'].nunique()}")

    verdict = G1 and G2 and G3
    print("\n" + "=" * 64)
    print(f"GATE {'PASSED' if verdict else 'FAILED'}  (G1={G1}, G2={G2}, G3={G3})")
    print("=" * 64)
    print("-> " + ("proceed to coverage_full_study.py (one-shot confirmatory)."
                   if verdict else "do NOT run the confirmatory; report per section 9."))
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
