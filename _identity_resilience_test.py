"""
Construct-native resilience / offensive-identity test.

Claim tested (ONLY this — not PPP, not cost, not shaper 'quality'):
  When a Shaper/Hub leaves the floor, does the team's offensive association
  structure change more than when a usage-matched non-organizer leaves?

Outcomes (identity, not scoring):
  D_A  = 1 - cos(A_on, A_off)   association-matrix displacement
  D_pi = 0.5 * L1(pi_on, pi_off)  play-type mix total variation

Also: does season LOO load L(p) predict D_A? (construct check)

Comparisons:
  SHAPER vs ROLE OCCUPANT / SPECIALIST (usage-matched)
  SHAPER vs TERMINAL SCORER (usage-matched)
  PRIMARY ORGANIZER reported as positive-control reference (not the claim)

Run: python _identity_resilience_test.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from wiring_gate import PLAY_TYPES_8, OUT_DIR, IU, assoc_O, null_O_stats, standardize
from _connector_player_onoff_mechanism import (
    build_shaper_universe,
    pull_possessions_by_id,
    player_on_mask_id,
    IDCOLS,
)

MIN_ON = 400
MIN_OFF = 400
N_NULL = 80
SEED = 17
USAGE_CALIPER = 0.03  # absolute usg fraction (graph u) or usage points/100


def A_of(poss, rng):
    v = poss[
        poss.initiator_id.notna()
        & (poss.initiator_id.astype(str) != "")
        & (poss.initiator_id.astype(str) != "None")
    ]
    if v.empty:
        return None
    piv = (
        v.groupby(["initiator_id", "ptype"]).size()
        .unstack(fill_value=0)
        .reindex(columns=PLAY_TYPES_8, fill_value=0)
    )
    B = piv.to_numpy(dtype=np.int64)
    B = B[B.sum(axis=1) > 0]
    if B.shape[0] < 3 or B.sum() < 200:
        return None
    E, sd, _ = null_O_stats(B, rng, N_NULL)
    return standardize(assoc_O(B), E, sd)


def pi_of(poss):
    vc = poss["ptype"].value_counts(normalize=True)
    return np.array([float(vc.get(pt, 0.0)) for pt in PLAY_TYPES_8])


def identity_metrics(on_df, off_df, rng):
    pi_on, pi_off = pi_of(on_df), pi_of(off_df)
    d_pi = 0.5 * float(np.abs(pi_on - pi_off).sum())  # TV distance in [0,1]
    # cosine of mix
    denom = np.linalg.norm(pi_on) * np.linalg.norm(pi_off)
    cos_pi = float(np.dot(pi_on, pi_off) / denom) if denom > 0 else np.nan
    d_pi_cos = 1.0 - cos_pi if np.isfinite(cos_pi) else np.nan

    A_on, A_off = A_of(on_df, rng), A_of(off_df, rng)
    if A_on is None or A_off is None:
        return dict(D_A=np.nan, D_pi=d_pi, D_pi_cos=d_pi_cos, ok_A=False)
    v_on, v_off = A_on[IU], A_off[IU]
    den = np.linalg.norm(v_on) * np.linalg.norm(v_off)
    d_a = 1.0 - float(np.dot(v_on, v_off) / den) if den > 0 else np.nan
    return dict(D_A=d_a, D_pi=d_pi, D_pi_cos=d_pi_cos, ok_A=True)


def compute_panel(rot: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    ts_cache = {}
    n = len(rot)
    for i, (_, r) in enumerate(rot.iterrows(), 1):
        team, yy = r["team"], int(r["yy"])
        key = (team, yy)
        if key not in ts_cache:
            ts_cache[key] = df[(df.team == team) & (df.yy.astype(int) == yy)].copy()
        sub = ts_cache[key]
        pid = str(r["player_id"])
        on = player_on_mask_id(sub, pid)
        n_on, n_off = int(on.sum()), int((~on).sum())
        if n_on < MIN_ON or n_off < MIN_OFF:
            continue
        mets = identity_metrics(sub.loc[on], sub.loc[~on], rng)
        rows.append(dict(
            player_id=pid,
            player=r["player"],
            full_name=r.get("full_name", r["player"]),
            team=team,
            season=r["season"],
            team_season=f"{team} {r['season']}",
            role=r["role_classification"],
            usage=float(r["usage"]),
            structural_influence=float(r["structural_influence"]),
            influence_residual=float(r["influence_residual"]),
            graph_u=float(r["u"]) if np.isfinite(r.get("u", np.nan)) else np.nan,
            n_on=n_on,
            n_off=n_off,
            min_half=min(n_on, n_off),
            **mets,
        ))
        if i % 50 == 0:
            print(f"  {i}/{n} scanned; kept {len(rows)}", flush=True)
    return pd.DataFrame(rows)


def usage_match(treat: pd.DataFrame, control: pd.DataFrame, caliper=USAGE_CALIPER):
    """1:1 nearest-neighbor on usage; prefer same team-season; no replacement."""
    ctrl = control.copy().reset_index(drop=True)
    pairs = []
    for _, t in treat.sort_values("usage").iterrows():
        if ctrl.empty:
            break
        cands = ctrl[
            ~((ctrl.player_id == t.player_id) & (ctrl.team_season == t.team_season))
        ].copy()
        if cands.empty:
            continue
        same = cands[cands.team_season == t.team_season]
        pool = same if len(same) else cands
        pool = pool.assign(du=(pool.usage - t.usage).abs())
        pool = pool[pool.du <= caliper]
        if pool.empty:
            continue
        best = pool.nsmallest(1, "du").iloc[0]
        pairs.append(dict(
            t_id=t.player_id, t_name=t.full_name, t_ts=t.team_season,
            t_usage=t.usage, t_L=t.structural_influence, t_DA=t.D_A, t_Dpi=t.D_pi,
            c_id=best.player_id, c_name=best.full_name, c_ts=best.team_season,
            c_usage=best.usage, c_L=best.structural_influence, c_DA=best.D_A, c_Dpi=best.D_pi,
            c_role=best.role, du=float(best.du),
            same_ts=int(best.team_season == t.team_season),
            d_DA=float(t.D_A - best.D_A) if np.isfinite(t.D_A) and np.isfinite(best.D_A) else np.nan,
            d_Dpi=float(t.D_pi - best.D_pi),
        ))
        ctrl = ctrl[~((ctrl.player_id == best.player_id) & (ctrl.team_season == best.team_season))]
    return pd.DataFrame(pairs)


def class_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role, g in panel.groupby("role"):
        gA = g[np.isfinite(g.D_A)]
        rows.append(dict(
            role=role, n=len(g), n_A=len(gA),
            mean_DA=gA.D_A.mean(), median_DA=gA.D_A.median(),
            mean_Dpi=g.D_pi.mean(), median_Dpi=g.D_pi.median(),
            mean_L=g.structural_influence.mean(),
            mean_usage=g.usage.mean(),
            mean_min_half=g.min_half.mean(),
        ))
    return pd.DataFrame(rows).sort_values("mean_DA", ascending=False)


def regress_DA_on_L(panel: pd.DataFrame) -> dict:
    d = panel[np.isfinite(panel.D_A)].copy()
    # residualize D_A and L on log min_half + usage (sample-size / minutes confound)
    X = np.column_stack([
        np.ones(len(d)),
        np.log(d.min_half.to_numpy(float)),
        d.usage.to_numpy(float),
    ])
    for col in ["D_A", "structural_influence"]:
        y = d[col].to_numpy(float)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        d[col + "_r"] = y - X @ b
    r, p = stats.pearsonr(d["structural_influence_r"], d["D_A_r"])
    r_raw, p_raw = stats.pearsonr(d["structural_influence"], d["D_A"])
    # partial: within role
    out = dict(r_raw=r_raw, p_raw=p_raw, r_adj=r, p_adj=p, n=len(d))
    # shapers only
    c = d[d.role == "SHAPER"]
    if len(c) >= 20:
        out["r_adj_conn"] = stats.pearsonr(c["structural_influence_r"], c["D_A_r"])[0]
        out["p_adj_conn"] = stats.pearsonr(c["structural_influence_r"], c["D_A_r"])[1]
    return out


def welch(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return dict(mean_a=np.nan, mean_b=np.nan, diff=np.nan, t=np.nan, p=np.nan, n_a=len(a), n_b=len(b))
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return dict(
        mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
        diff=float(np.mean(a) - np.mean(b)),
        t=float(t), p=float(p), n_a=len(a), n_b=len(b),
    )


def paired_test(pairs: pd.DataFrame, col="d_DA"):
    x = pairs[col].to_numpy(float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return dict(n=len(x), mean_d=np.nan, t=np.nan, p=np.nan)
    t, p = stats.ttest_1samp(x, 0.0)
    return dict(n=len(x), mean_d=float(np.mean(x)), t=float(t), p=float(p),
                median_d=float(np.median(x)),
                pos_frac=float((x > 0).mean()))


def build_report(panel, summary, pairs_ro, pairs_ts, Lfit, unc_ro, unc_ts) -> str:
    lines = [
        "# Offensive-identity resilience test (construct-native)",
        "",
        "**Claim tested:** Shaper absence displaces team offensive *identity* "
        "(association structure / play-type mix) more than usage-matched non-organizer absence.",
        "",
        "**Not claimed:** PPP value, cost efficiency, shaper quality, defense.",
        "",
        f"Sample: player-seasons with ≥{MIN_ON} possessions on and off. "
        f"N kept = {len(panel)} (D_A available for {int(np.isfinite(panel.D_A).sum())}).",
        "",
        "## 1. Class means (unmatched)",
        "",
        "| role | n | mean D_A | median D_A | mean D_π | mean L | mean usage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r.role} | {int(r.n)} | {r.mean_DA:.4f} | {r.median_DA:.4f} | "
            f"{r.mean_Dpi:.4f} | {r.mean_L:.4f} | {r.mean_usage:.3f} |"
        )

    lines += [
        "",
        "PRIMARY ORGANIZER is a positive-control reference (high usage + high residual influence).",
        "The licensed comparison for the claim excludes organizers.",
        "",
        "## 2. Unmatched contrasts (shapers vs others)",
        "",
    ]
    for name, u in [("Shaper vs Role Occupant", unc_ro),
                    ("Shaper vs Terminal Scorer", unc_ts)]:
        lines.append(
            f"- **{name} (D_A):** "
            f"Δ = {u['diff']:+.4f} "
            f"(conn {u['mean_a']:.4f} vs other {u['mean_b']:.4f}; "
            f"n={u['n_a']}/{u['n_b']}; t={u['t']:.2f}, p={u['p']:.3g})"
        )

    lines += ["", "## 3. Usage-matched pairs (primary test)", ""]
    for label, pairs in [
        ("Shaper vs Role Occupant", pairs_ro),
        ("Shaper vs Terminal Scorer", pairs_ts),
    ]:
        if pairs is None or len(pairs) == 0:
            lines.append(f"### {label}: no pairs within caliper {USAGE_CALIPER}")
            continue
        pA = paired_test(pairs, "d_DA")
        pPi = paired_test(pairs, "d_Dpi")
        lines += [
            f"### {label}",
            f"- Pairs: {len(pairs)} (same team-season: {int(pairs.same_ts.sum())})",
            f"- Mean usage gap: {pairs.du.mean():.4f}",
            f"- **Paired Δ D_A (conn − control): {pA['mean_d']:+.4f}** "
            f"(median {pA.get('median_d', float('nan')):+.4f}; "
            f"P(conn>ctrl)={100*pA.get('pos_frac', float('nan')):.1f}%; "
            f"t={pA['t']:.2f}, p={pA['p']:.3g}, n={pA['n']})",
            f"- Paired Δ D_π: {pPi['mean_d']:+.4f} "
            f"(t={pPi['t']:.2f}, p={pPi['p']:.3g})",
            "",
        ]

    lines += [
        "## 4. Does L(p) predict identity break D_A?",
        "",
        f"- Raw corr(L, D_A): r={Lfit['r_raw']:+.3f} (p={Lfit['p_raw']:.3g}, n={Lfit['n']})",
        f"- Residualized for log(min on/off) + usage: r={Lfit['r_adj']:+.3f} "
        f"(p={Lfit['p_adj']:.3g})",
    ]
    if "r_adj_conn" in Lfit:
        lines.append(
            f"- Within shapers only (adj): r={Lfit['r_adj_conn']:+.3f} "
            f"(p={Lfit['p_adj_conn']:.3g})"
        )
    lines += [
        "",
        "If L measures dependence, higher L should mean larger D_A when the player is off.",
        "",
        "## 5. Verdict",
        "",
    ]

    # decide from matched RO comparison (main)
    main = paired_test(pairs_ro, "d_DA") if len(pairs_ro) else dict(p=1, mean_d=0, pos_frac=0.5)
    if main["p"] < 0.05 and main["mean_d"] > 0:
        lines += [
            "**SUPPORTED (identity value):** Shaper absence displaces offensive "
            "association structure more than usage-matched role-occupant absence.",
            "Licensed claim: shapers matter for **maintaining offensive identity under "
            "substitution** — not for points per possession.",
        ]
    elif main["p"] < 0.05 and main["mean_d"] < 0:
        lines += [
            "**REJECTED / REVERSED:** Matched role occupants displace identity *more* "
            "than shapers. Do not claim identity-resilience value for the shaper class.",
        ]
    else:
        # check L prediction as softer support
        if Lfit["p_adj"] < 0.05 and Lfit["r_adj"] > 0:
            lines += [
                "**PARTIAL:** Class contrast vs matched role occupants is not significant, "
                "but L(p) still predicts D_A after usage/minute controls — dependence tracks "
                "identity break continuously. Class-level shaper claim is weak; "
                "load-level resilience claim has support.",
            ]
        else:
            lines += [
                "**NOT SUPPORTED:** No clear evidence that shaper absence uniquely "
                "threatens offensive identity versus usage-matched non-organizers. "
                "Do not claim resilience/identity value for the class from this test.",
            ]

    # top examples
    conn = panel[panel.role == "SHAPER"].nlargest(8, "D_A")
    lines += ["", "### Highest D_A shapers (identity most disrupted when off)", ""]
    for _, r in conn.iterrows():
        lines.append(
            f"- {r.full_name} ({r.team_season}): D_A={r.D_A:.3f}, D_π={r.D_pi:.3f}, "
            f"L={r.structural_influence:.3f}, usg={r.usage:.3f}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    t0 = time.time()
    print("building role universe...", flush=True)
    _, rot = build_shaper_universe()
    print(rot.role_classification.value_counts().to_string(), flush=True)

    print("loading possessions...", flush=True)
    df = pull_possessions_by_id()
    df = df[df.pts.notna()].copy()
    for c in IDCOLS:
        df[c] = df[c].astype(str)
    df["initiator_id"] = df["initiator_id"].astype(str)

    print("computing on/off identity metrics...", flush=True)
    panel = compute_panel(rot, df)
    panel.to_csv(OUT_DIR / "identity_resilience_panel.csv", index=False)
    print(f"kept {len(panel)} player-seasons", flush=True)

    summary = class_summary(panel)
    summary.to_csv(OUT_DIR / "identity_resilience_class_means.csv", index=False)

    conn = panel[panel.role == "SHAPER"]
    ro = panel[panel.role == "ROLE OCCUPANT / SPECIALIST"]
    ts = panel[panel.role == "TERMINAL SCORER"]
    org = panel[panel.role == "PRIMARY ORGANIZER"]

    unc_ro = welch(conn.D_A, ro.D_A)
    unc_ts = welch(conn.D_A, ts.D_A)
    unc_org = welch(org.D_A, conn.D_A)
    print("unmatched conn vs RO", unc_ro, flush=True)
    print("unmatched conn vs TS", unc_ts, flush=True)
    print("unmatched ORG vs conn", unc_org, flush=True)

    print("usage matching...", flush=True)
    pairs_ro = usage_match(conn[np.isfinite(conn.D_A)], ro[np.isfinite(ro.D_A)])
    pairs_ts = usage_match(conn[np.isfinite(conn.D_A)], ts[np.isfinite(ts.D_A)])
    pairs_ro.to_csv(OUT_DIR / "identity_resilience_pairs_vs_roleoccupant.csv", index=False)
    pairs_ts.to_csv(OUT_DIR / "identity_resilience_pairs_vs_terminal.csv", index=False)
    print(f"pairs vs RO: {len(pairs_ro)}; vs TS: {len(pairs_ts)}", flush=True)
    if len(pairs_ro):
        print("paired DA", paired_test(pairs_ro, "d_DA"), flush=True)
        print("paired Dpi", paired_test(pairs_ro, "d_Dpi"), flush=True)

    Lfit = regress_DA_on_L(panel)
    print("L -> D_A", Lfit, flush=True)

    report = build_report(panel, summary, pairs_ro, pairs_ts, Lfit, unc_ro, unc_ts)
    # add organizer reference line
    report += (
        f"\n### Reference: Organizer vs Shaper (unmatched D_A)\n"
        f"Organizers mean D_A={unc_org['mean_a']:.4f} vs shapers {unc_org['mean_b']:.4f} "
        f"(Δ={unc_org['diff']:+.4f}, p={unc_org['p']:.3g}). "
        f"Organizers are high-usage positive controls, not part of the licensed claim.\n"
    )
    (OUT_DIR / "identity_resilience_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
