"""
PORTABILITY STUDY — FEASIBILITY GATE (§7.1)
===========================================
Feasibility ONLY. Deliberately does NOT compute median Δ or the self-vs-decoy
contrast (peeking would create optional-stopping). Checks:

  G1 sample exists    : >=25 movers with valid two-sided signatures.
  G2 decoy pools       : >=80% of movers have >=3 profile-matched decoys.
  G3 not vacuous       : |Spearman(cos_profile, cos_signature)| < 0.90 across random
                         cross-player pairs (control not collinear with the object).

Any single failure => do NOT run the full study.
Run: python portability_gate.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from portability_common import build_signature_cache, cossim, SIG_COLS, PROF_COLS, OUT_DIR

K = 5
MIN_DECOYS = 3
SEED = 17


def build_movers(sig, max_gap=0):
    """Chronologically ordered franchise-change events, keyed by NBA person_id
    (`player_id`), not abbreviated display name (names collide across players).

    A mover is a pair of the player's consecutive included stints, ordered by
    `anchor` (min game_id of that team-stint), where the franchise changes.
    Adjacent means no other included stint for this player_id sits between them.
    Mid-season trades can share the same yy label; those pairs are valid when
    they are chronological neighbors.

    gap_seasons = max(0, yy2 - yy1 - 1) is a covariate (0 for same-yy mid-season
    adjacency or clean yy→yy+1). max_gap bounds that gap (0 = confirmatory
    default). An intervening stint below the signature volume floor leaves no
    row in `sig` and is invisible here; named exemplars should be checked
    against transaction records.
    """
    sig_mat = {i: sig.loc[i, SIG_COLS].to_numpy(float) for i in sig.index}
    prof_mat = {i: sig.loc[i, PROF_COLS].to_numpy(float) for i in sig.index}
    movers = []
    for pid, g in sig.groupby("player_id"):
        g = g.sort_values("anchor")
        stints = list(g.itertuples(index=True))
        for a, b in zip(stints[:-1], stints[1:]):
            if a.tri == b.tri:
                continue  # same franchise across these two included stints -- not a move
            # a, b are chronological neighbors; gap_seasons is a covariate
            gap = max(0, int(b.yy) - int(a.yy) - 1)
            if gap > max_gap:
                continue
            i1, i2 = a.Index, b.Index
            key2 = b.key
            # decoys: same destination team-season, profile-nearest, exclude self
            cohort = sig[(sig["key"] == key2) & (sig["player_id"] != pid)]
            if len(cohort):
                sims = [(j, cossim(prof_mat[i2], prof_mat[j])) for j in cohort.index]
                sims.sort(key=lambda t: -t[1])
                decoys = [j for j, _ in sims[:K]]
            else:
                decoys = []
            mover_id = f"{pid}_{a.tri}{a.yy}_{b.tri}{b.yy}"
            movers.append(dict(mover_id=mover_id, player_id=pid, player=b.player,
                               f1=a.tri, f2=b.tri, yy1=int(a.yy), yy2=int(b.yy),
                               gap_seasons=gap, i1=i1, i2=i2, dest_key=key2,
                               n_decoys=len(decoys), decoys=decoys))
    return movers, sig_mat, prof_mat


PREREG_SEASONS = (22, 23, 24)  # pre-registered 2022-23..2024-25 window; 2025-26 excluded
                                # from all confirmatory portability tests (movers whose
                                # origin or destination stint falls outside this window
                                # are dropped by construction, since build_movers only
                                # pairs stints present in `sig`).


def main():
    sig = build_signature_cache().reset_index(drop=True)
    sig = sig[sig["yy"].isin(PREREG_SEASONS)].reset_index(drop=True)
    print(f"\nplayer-signatures: {len(sig)}  team-seasons: {sig['key'].nunique()}  "
          f"players: {sig['player'].nunique()}  (restricted to pre-registered seasons "
          f"{PREREG_SEASONS})")

    movers, sig_mat, prof_mat = build_movers(sig)
    n_movers = len(movers)
    g1 = n_movers >= 25
    print(f"\nG1 sample exists: {n_movers} movers (need >=25) -> {'PASS' if g1 else 'FAIL'}")

    enough = sum(1 for m in movers if m["n_decoys"] >= MIN_DECOYS)
    frac = enough / n_movers if n_movers else 0.0
    g2 = frac >= 0.80
    print(f"G2 decoy pools: {enough}/{n_movers} movers have >={MIN_DECOYS} decoys "
          f"({frac:.2f}, need >=0.80) -> {'PASS' if g2 else 'FAIL'}")

    # G3 vacuity: across random cross-player pairs, is signature-sim determined by profile-sim?
    rng = np.random.default_rng(SEED)
    idx = sig.index.to_numpy()
    cps, css = [], []
    for _ in range(4000):
        a, b = rng.choice(idx, 2, replace=False)
        cps.append(cossim(prof_mat[a], prof_mat[b]))
        css.append(cossim(sig_mat[a], sig_mat[b]))
    rho = pd.Series(cps).corr(pd.Series(css), method="spearman")
    g3 = abs(rho) < 0.90
    print(f"G3 not vacuous: |Spearman(cos_profile,cos_signature)|={abs(rho):.3f} "
          f"(need <0.90) -> {'PASS' if g3 else 'FAIL'}")

    gate = g1 and g2 and g3
    print("\n" + "=" * 64)
    print("PORTABILITY FEASIBILITY GATE — VERDICT")
    print("=" * 64)
    print(f"  G1 sample exists : {'PASS' if g1 else 'FAIL'}  ({n_movers} movers)")
    print(f"  G2 decoy pools   : {'PASS' if g2 else 'FAIL'}  ({frac:.2f})")
    print(f"  G3 not vacuous   : {'PASS' if g3 else 'FAIL'}  (|rho|={abs(rho):.3f})")
    verdict = ("GATE PASSES — movers + decoys exist and the signature is separable from the "
               "play-type profile; proceed to confirmatory"
               if gate else
               "GATE FAILS — do NOT run the full study; record the reason (§9)")
    print(f"\n  {verdict}")
    print("=" * 64)

    with open(OUT_DIR / "portability_gate_result.txt", "w", encoding="utf-8") as f:
        f.write("PORTABILITY FEASIBILITY GATE\n\n")
        f.write(f"movers={n_movers} decoy>=3 frac={frac:.2f} vacuity_rho={abs(rho):.3f}\n")
        f.write(f"G1 sample exists : {'PASS' if g1 else 'FAIL'}\n")
        f.write(f"G2 decoy pools   : {'PASS' if g2 else 'FAIL'}\n")
        f.write(f"G3 not vacuous   : {'PASS' if g3 else 'FAIL'}\n")
        f.write(f"\nVERDICT: {verdict}\n")
    print(f"\nWrote {OUT_DIR / 'portability_gate_result.txt'}")


if __name__ == "__main__":
    main()
