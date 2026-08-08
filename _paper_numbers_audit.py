"""
PAPER NUMBERS AUDIT - Structural Influence (paper.tex)
=================================================
Recomputes locked numeric claims from output/ (no Neo4j).

Usage:
  python _paper_numbers_audit.py            # verify + write PAPER_NUMBERS_SHEET.md
  python _paper_numbers_audit.py --check    # verify only (exit 1 on FAIL)
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, ttest_1samp

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
SHEET = ROOT / "PAPER_NUMBERS_SHEET.md"

TOL = {
    "exact": 0.0,
    "int": 0.0,
    "round1": 0.06,
    "round2": 0.015,
    "round3": 0.0055,
    "pct_tenth": 0.15,
    "soft": 0.05,
}


class Claim:
    __slots__ = ("section", "claim", "paper", "actual", "tol", "script",
                 "artifact", "note", "ok")

    def __init__(self, section, claim, paper, actual, tol, script, artifact,
                 note=""):
        self.section = section
        self.claim = claim
        self.paper = paper
        self.actual = actual
        self.tol = tol
        self.script = script
        self.artifact = artifact
        self.note = note
        if paper is None or actual is None or (
            isinstance(actual, float) and not math.isfinite(actual)
        ):
            self.ok = False
        elif isinstance(paper, (int, float)) and isinstance(actual, (int, float)):
            self.ok = abs(float(actual) - float(paper)) <= tol + 1e-12
        else:
            self.ok = str(paper) == str(actual)


def _sign_test_p(n_pos, n):
    return float(binomtest(int(n_pos), int(n), 0.5).pvalue)


def _mde80(se_pts100):
    return (1.959964 + 0.841621) * float(se_pts100)


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------
def claims_data():
    out = []
    pq = OUT / "role_composition_possessions.parquet"
    n_poss = int(len(pd.read_parquet(pq))) if pq.exists() else None
    out.append(Claim(
        "§Data", "corpus possessions (~6.5e5)",
        6.5e5, float(n_poss) if n_poss else None, 2.0e4,
        "role_composition_playmix.py",
        "role_composition_possessions.parquet",
        f"exact n={n_poss}" if n_poss else "missing",
    ))
    cov = pd.read_csv(OUT / "coverage_full_study.csv")
    out.append(Claim(
        "§Data / §Efficiency", "coverage primary N possessions",
        420993, int(cov.iloc[0]["n"]), TOL["int"],
        "coverage_full_study.py", "coverage_full_study.csv",
    ))
    return out


def claims_taxonomy():
    s = pd.read_csv(OUT / "structural_role_summary.csv")
    out = [Claim("§6.1", "rotation pool n", 924, int(s["n"].sum()),
                 TOL["int"], "structural_role_taxonomy.py",
                 "structural_role_summary.csv")]
    expect = {
        "PRIMARY ORGANIZER": dict(n=96, pct=10.4, u=0.244, L=0.094,
                                  resid=0.054, ppg=18.4, mpg=29.6),
        "SHAPER": dict(n=103, pct=11.1, u=0.145, L=0.078,
                                resid=0.062, ppg=9.0, mpg=23.6),
        "TERMINAL SCORER": dict(n=368, pct=39.8, u=0.233, L=0.014,
                                resid=-0.019, ppg=16.6, mpg=28.6),
        "ROLE OCCUPANT / SPECIALIST": dict(n=357, pct=38.6, u=0.147, L=0.006,
                                           resid=-0.009, ppg=8.2, mpg=22.8),
    }
    for _, row in s.iterrows():
        e = expect[row["role"]]
        for key, paper_v, col, tol in [
            ("n", e["n"], "n", TOL["int"]),
            ("% pool", e["pct"], "pct_of_pool", TOL["round2"]),
            ("usage", e["u"], "avg_usage", TOL["round3"]),
            ("influence", e["L"], "avg_structural_influence", TOL["round3"]),
            ("resid influence", e["resid"], "avg_influence_residual", TOL["round3"]),
            ("ppg", e["ppg"], "avg_ppg", TOL["round2"]),
            ("mpg", e["mpg"], "avg_mpg", TOL["round2"]),
        ]:
            out.append(Claim(
                "§6.1 Table taxonomy", f"{row['role']} {key}",
                paper_v, float(row[col]), tol,
                "structural_role_taxonomy.py",
                "structural_role_summary.csv",
            ))
    return out


def claims_exemplars():
    clf = pd.read_csv(OUT / "structural_role_classification.csv")
    # Bold / named exemplars from Table tab:exemplars
    paper_rows = [
        ("S. Gilgeous-Alexander", "2023-24", 0.317, 0.308, 0.216),
        ("A. Davis", "2023-24", 0.261, 0.168, 0.105),
        ("N. Jokić", "2024-25", 0.285, 0.145, 0.067),
        ("B. Adebayo", "2023-24", 0.247, 0.211, 0.166),
        ("R. Gobert", "2023-24", 0.152, 0.620, 0.598),
        ("D. Gafford", "2022-23", 0.146, 0.411, 0.397),
        ("J. Poeltl", "2023-24", 0.160, 0.217, 0.203),
        ("J. Allen", "2024-25", 0.156, 0.110, 0.088),
        ("L. James", "2023-24", 0.285, 0.013, -0.054),
        ("J. Tatum", "2022-23", 0.319, 0.009, -0.090),
        ("J. Hart", "2023-24", 0.133, 0.003, -0.022),
    ]
    out = []
    for player, season, u, L, resid in paper_rows:
        hit = clf[(clf["player"] == player) & (clf["season"] == season)]
        if hit.empty and "Joki" in player:
            hit = clf[clf["player"].str.contains("Joki", na=False)
                      & (clf["season"] == season)]
        if hit.empty:
            out.append(Claim("§6.1 exemplars", f"{player} {season} present",
                             1, 0, 0, "structural_role_taxonomy.py",
                             "structural_role_classification.csv", "MISSING"))
            continue
        r = hit.iloc[0]
        label = f"{player} {season}"
        out.append(Claim("§6.1 exemplars", f"{label} usage",
                         u, float(r["usage"]), TOL["round3"],
                         "structural_role_taxonomy.py",
                         "structural_role_classification.csv"))
        out.append(Claim("§6.1 exemplars", f"{label} influence",
                         L, float(r["structural_influence"]), TOL["round3"],
                         "structural_role_taxonomy.py",
                         "structural_role_classification.csv"))
        out.append(Claim("§6.1 exemplars", f"{label} resid",
                         resid, float(r["influence_residual"]), TOL["round3"],
                         "structural_role_taxonomy.py",
                         "structural_role_classification.csv"))
    return out


def claims_ortho():
    o = pd.read_csv(OUT / "fragility_orthogonality.csv")
    m = o[o["net_rating"].notna() & o["Lcos"].notna()
          & o["u_resid"].notna() & o["usg_pct"].notna()].copy()
    out = [Claim("§6.2", "orthogonality n", 1283, len(m), TOL["int"],
                 "fragility_orthogonality.py",
                 "fragility_orthogonality.csv")]
    paper = {
        "usg_pct": (+0.270, -0.261),
        "ppg": (+0.436, -0.264),
        "apg": (+0.278, -0.255),
        "ast_pct": (+0.106, -0.190),
        "touches": (+0.376, -0.253),
        "front_ct_touches": (+0.424, -0.238),
        "net_rating": (+0.203, -0.018),
    }
    for col, (p_load, p_resid) in paper.items():
        r_load = spearmanr(m["Lcos"], m[col], nan_policy="omit").correlation
        r_resid = spearmanr(m["u_resid"], m[col], nan_policy="omit").correlation
        out.append(Claim("§6.2 Table ortho", f"ρ(influence,{col})",
                         p_load, float(r_load), TOL["round3"],
                         "fragility_orthogonality.py",
                         "fragility_orthogonality.csv"))
        out.append(Claim("§6.2 Table ortho", f"ρ(resid,{col})",
                         p_resid, float(r_resid), TOL["round3"],
                         "fragility_orthogonality.py",
                         "fragility_orthogonality.csv"))
    return out


def claims_behavior():
    pm = pd.read_csv(OUT / "role_composition_playmix.csv")
    out = []
    conn = pm[pm["role_class"] == "SHAPER"].set_index("play_type")
    paper_conn = {
        "CUT": 1.35, "PNR": 0.98, "SPOT_UP": -1.25, "DRIVE": -0.54,
    }
    n_ts = int(conn["n_ts"].iloc[0])
    out.append(Claim("§6.3", "shaper strata team-seasons", 73, n_ts,
                     TOL["int"], "role_composition_playmix.py",
                     "role_composition_playmix.csv"))
    for pt, pv in paper_conn.items():
        out.append(Claim("§6.3", f"shaper Δshare {pt} (pp)",
                         pv, float(conn.loc[pt, "mean_delta_pp"]), TOL["round2"],
                         "role_composition_playmix.py",
                         "role_composition_playmix.csv"))
    org = pm[pm["role_class"] == "ORGANIZER"].set_index("play_type")
    out.append(Claim("§6.3", "organizer strata team-seasons", 58,
                     int(org["n_ts"].iloc[0]), TOL["int"],
                     "role_composition_playmix.py",
                     "role_composition_playmix.csv"))
    out.append(Claim("§6.3", "organizer Δshare POST_UP (pp)", 2.33,
                     float(org.loc["POST_UP", "mean_delta_pp"]), TOL["round2"],
                     "role_composition_playmix.py",
                     "role_composition_playmix.csv"))
    out.append(Claim("§6.3", "organizer Δshare PNR (pp)", 0.96,
                     float(org.loc["PNR", "mean_delta_pp"]), TOL["round2"],
                     "role_composition_playmix.py",
                     "role_composition_playmix.csv"))

    rr = pd.read_csv(OUT / "role_route_graph.csv")
    c = rr[rr["focal"] == "SHAPER"].iloc[0]
    out.append(Claim(
        "§6.3", "shaper Δentropy (nonfocal) near 0",
        0.0, float(c["d_entropy_nonfocal"]), 0.10,
        "role_route_graph.py", "role_route_graph.csv",
        f"artifact d_entropy_nonfocal={c['d_entropy_nonfocal']:+.3f}",
    ))
    return out


def claims_efficiency():
    cov = pd.read_csv(OUT / "coverage_full_study.csv")
    out = []
    # Covered rows: pts/100, t, CI lo/hi (pts/100)
    expect = [
        ("M1 PRIMARY", 0.42, 0.53, -1.15, 1.90),
        ("WITH play-type", 0.44, 0.53, -1.29, 2.04),
        ("talent REMOVED", 0.43, 0.54, -1.15, 1.94),
        (">=4/5", 0.26, 0.37, -1.23, 1.67),
    ]
    for i, (tag, pts, t, lo, hi) in enumerate(expect):
        r = cov.iloc[i]
        out.append(Claim("§6.4 Table coverage", f"{tag} pts/100", pts,
                         float(r["coef_COVERED_pts100"]), TOL["round2"],
                         "coverage_full_study.py", "coverage_full_study.csv"))
        out.append(Claim("§6.4 Table coverage", f"{tag} t", t,
                         float(r["t_COVERED"]), TOL["round2"],
                         "coverage_full_study.py", "coverage_full_study.csv"))
        out.append(Claim("§6.4 Table coverage", f"{tag} CI lo", lo,
                         float(r["ci_lo_ts"]) * 100, TOL["round2"],
                         "coverage_full_study.py", "coverage_full_study.csv"))
        out.append(Claim("§6.4 Table coverage", f"{tag} CI hi", hi,
                         float(r["ci_hi_ts"]) * 100, TOL["round2"],
                         "coverage_full_study.py", "coverage_full_study.csv"))

    out.append(Claim("§6.4", "n_organizer pts/100", 2.35,
                     float(cov.iloc[0]["coef_n_ORGANIZER_pts100"]),
                     TOL["round2"], "coverage_full_study.py",
                     "coverage_full_study.csv"))
    out.append(Claim("§6.4", "n_organizer t", 4.62,
                     float(cov.iloc[0]["t_n_ORGANIZER"]),
                     TOL["round2"], "coverage_full_study.py",
                     "coverage_full_study.csv"))
    se = float(cov.iloc[0]["cluster_se"]) * 100
    out.append(Claim("§6.4", "MDE(80%) pts/100 ≈2.2", 2.2,
                     float(cov.iloc[0]["mde80_pts100"]), 0.05,
                     "coverage_full_study.py", "coverage_full_study.csv",
                     f"SE={se:.2f}; formula MDE={_mde80(se):.2f}"))

    h = pd.read_csv(OUT / "hasconn_full_study.csv").iloc[0]
    out.append(Claim("§6.4", "HasShaper pts/100", 0.00,
                     float(h["coef_HasShaper_pts100"]), TOL["round2"],
                     "hasconn_full_study.py", "hasconn_full_study.csv"))
    out.append(Claim("§6.4", "HasShaper t", 0.00,
                     float(h["t_HasShaper"]), TOL["round2"],
                     "hasconn_full_study.py", "hasconn_full_study.csv"))
    return out


def claims_connector_tiers():
    """Exploratory Table tab:shaper-tiers (M1 primary row)."""
    t = pd.read_csv(OUT / "connector_tier_full_study.csv").iloc[0]
    out = []
    rows = [
        ("HasHub", 1.10, 0.077, "coef_HasHub_pts100", "p_HasHub"),
        ("HasMargBig", 1.70, 0.025, "coef_HasMargBig_pts100", "p_HasMargBig"),
        ("HasMargWing", 1.29, 0.122, "coef_HasMargWing_pts100", "p_HasMargWing"),
    ]
    for name, pts, p, c_pts, c_p in rows:
        out.append(Claim("§6.4.1 tiers", f"{name} pts/100", pts,
                         float(t[c_pts]), TOL["round2"],
                         "connector_tier_full_study.py",
                         "connector_tier_full_study.csv"))
        out.append(Claim("§6.4.1 tiers", f"{name} p", p,
                         float(t[c_p]), 0.005,
                         "connector_tier_full_study.py",
                         "connector_tier_full_study.csv"))
    return out


def claims_identity():
    means = pd.read_csv(OUT / "identity_resilience_class_means.csv")
    pairs = pd.read_csv(OUT / "identity_resilience_pairs_vs_roleoccupant.csv")
    out = []
    expect = {
        "PRIMARY ORGANIZER": (96, 0.214, 0.170, 0.244),
        "SHAPER": (102, 0.202, 0.151, 0.144),
        "TERMINAL SCORER": (361, 0.112, 0.094, 0.233),
        "ROLE OCCUPANT / SPECIALIST": (349, 0.102, 0.085, 0.148),
    }
    for _, row in means.iterrows():
        n, mda, med, u = expect[row["role"]]
        out.append(Claim("§6.4.2 identity", f"{row['role']} n",
                         n, int(row["n"]), TOL["int"],
                         "_identity_resilience_test.py",
                         "identity_resilience_class_means.csv"))
        out.append(Claim("§6.4.2 identity", f"{row['role']} mean DA",
                         mda, float(row["mean_DA"]), TOL["round3"],
                         "_identity_resilience_test.py",
                         "identity_resilience_class_means.csv"))
        out.append(Claim("§6.4.2 identity", f"{row['role']} median DA",
                         med, float(row["median_DA"]), TOL["round3"],
                         "_identity_resilience_test.py",
                         "identity_resilience_class_means.csv"))
        out.append(Claim("§6.4.2 identity", f"{row['role']} mean usage",
                         u, float(row["mean_usage"]), TOL["round3"],
                         "_identity_resilience_test.py",
                         "identity_resilience_class_means.csv"))

    d = pairs["d_DA"].dropna()
    out.append(Claim("§6.4.2 identity", "conn vs RO pairs n", 96, len(d),
                     TOL["int"], "_identity_resilience_test.py",
                     "identity_resilience_pairs_vs_roleoccupant.csv"))
    out.append(Claim("§6.4.2 identity", "ΔDA mean", 0.115, float(d.mean()),
                     TOL["round3"], "_identity_resilience_test.py",
                     "identity_resilience_pairs_vs_roleoccupant.csv"))
    t = float(ttest_1samp(d, 0.0).statistic)
    out.append(Claim("§6.4.2 identity", "ΔDA t", 6.35, t, 0.05,
                     "_identity_resilience_test.py",
                     "identity_resilience_pairs_vs_roleoccupant.csv"))
    return out


def claims_portability():
    txt = (OUT / "portability_full_result.txt").read_text(encoding="utf-8")
    gate = (OUT / "portability_gate_result.txt").read_text(encoding="utf-8")
    enr = pd.read_csv(OUT / "portability_players_enriched.csv")
    locked = pd.read_csv(OUT / "class_portability_locked.csv")
    out = []

    out.append(Claim("§6.5", "movers n", 294, len(enr), TOL["int"],
                     "portability_full_study.py",
                     "portability_players_enriched.csv"))
    m = re.search(r"dest_franchises=(\d+)", txt)
    n_dest = int(m.group(1)) if m else None
    out.append(Claim("§6.5", "destination franchises", 30, n_dest, TOL["int"],
                     "portability_full_study.py", "portability_full_result.txt"))
    out.append(Claim("§6.5", "median self_sim", 0.129,
                     float(np.median(enr["self_sim"])), TOL["round3"],
                     "portability_full_study.py",
                     "portability_players_enriched.csv"))
    out.append(Claim("§6.5", "median decoy_sim", 0.049,
                     float(np.median(enr["decoy_sim"])), TOL["round3"],
                     "portability_full_study.py",
                     "portability_players_enriched.csv"))
    out.append(Claim("§6.5", "median Δ", 0.056,
                     float(np.median(enr["delta"])), TOL["round3"],
                     "portability_full_study.py",
                     "portability_players_enriched.csv"))

    n_pos = int((enr["delta"] > 0).sum())
    out.append(Claim("§6.5", "Δ>0 count", 173, n_pos, TOL["int"],
                     "portability_full_study.py",
                     "portability_players_enriched.csv"))
    out.append(Claim("§6.5", "Δ>0 rate 58.8%", 58.8,
                     100.0 * n_pos / len(enr), TOL["pct_tenth"],
                     "portability_full_study.py",
                     "portability_players_enriched.csv"))
    sign_p = _sign_test_p(n_pos, len(enr))
    out.append(Claim("§6.5", "sign-test p vs 50%", 2.872e-3, sign_p, 5e-4,
                     "portability_full_study.py / scipy.binometest",
                     "computed", f"exact p={sign_p:.3e}"))

    m = re.search(r"vacuity_rho=([0-9.]+)", gate)
    rho = float(m.group(1)) if m else None
    out.append(Claim("§6.5", "|Spearman(profile,signature)| vacuity", 0.007,
                     rho, TOL["round3"], "portability_gate.py",
                     "portability_gate_result.txt"))

    m = re.search(r"dest-fr CI \[([+\-0-9.]+),([+\-0-9.]+)\]", txt)
    lo, hi = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    # paper player CI [+0.015,+0.096]; primary dest-fr CI in result file
    m2 = re.search(r"player CI \[([+\-0-9.]+),([+\-0-9.]+)\]", txt)
    if m2:
        lo, hi = float(m2.group(1)), float(m2.group(2))
    out.append(Claim("§6.5", "player CI lo", 0.015, lo, TOL["round3"],
                     "portability_full_study.py", "portability_full_result.txt"))
    out.append(Claim("§6.5", "player CI hi", 0.096, hi, TOL["round3"],
                     "portability_full_study.py", "portability_full_result.txt"))

    # class table
    expect_class = {
        "Organizer": (30, 0.134, 70.0),
        "Shaper": (24, 0.124, 58.3),
        "Terminal": (117, 0.083, 60.7),
        "Role": (123, 0.018, 54.5),
    }
    for _, row in locked.iterrows():
        n, med, pct = expect_class[row["role"]]
        out.append(Claim("§6.5 class portability", f"{row['role']} n",
                         n, int(row["N"]), TOL["int"],
                         "portability_players_report.py",
                         "class_portability_locked.csv"))
        out.append(Claim("§6.5 class portability", f"{row['role']} median Δ",
                         med, float(row["median_delta"]), TOL["round3"],
                         "portability_players_report.py",
                         "class_portability_locked.csv"))
        out.append(Claim("§6.5 class portability", f"{row['role']} % portable",
                         pct, float(row["pct_pos"]), TOL["round1"],
                         "portability_players_report.py",
                         "class_portability_locked.csv"))

    # Table tab:cases — shaper distribution endpoints
    cases = {
        "D. Gafford": dict(self_sim=0.86, delta=0.60),
        "D. Jones Jr.": dict(self_sim=0.21, delta=0.17),
        "D. Wright": dict(self_sim=-0.66, delta=-0.71),
    }
    for player, exp in cases.items():
        rows = enr[enr["player"] == player]
        if player == "D. Jones Jr.":
            # near-median row in paper: self_sim≈0.21
            rows = rows.iloc[(rows["self_sim"] - 0.21).abs().argsort()]
        if player == "D. Wright":
            rows = rows.iloc[(rows["self_sim"] + 0.66).abs().argsort()]
        if len(rows) == 0:
            out.append(Claim("§6.5 cases", f"{player} present", 1, 0, 0,
                             "portability_players_report.py",
                             "portability_players_enriched.csv", "MISSING"))
            continue
        r = rows.iloc[0]
        for k, pv in exp.items():
            out.append(Claim(
                "§6.5 cases", f"{player} {k}",
                pv, float(r[k]), TOL["round2"],
                "portability_players_report.py",
                "portability_players_enriched.csv",
            ))
    return out


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.empty(m, float)
    ranks[order] = np.arange(1, m + 1)
    q = p * m / ranks
    # enforce monotonicity from the largest p upward
    q_sorted = q[order]
    for i in range(m - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    out = np.empty(m, float)
    out[order] = np.minimum(q_sorted, 1.0)
    return out


def claims_travelfit():
    tf = pd.read_csv(OUT / "travel_fit_screen.csv")
    out = []
    out.append(Claim("§6.6", "travel×fit movers n", 294, len(tf), TOL["int"],
                     "travel_fit_screen.py", "travel_fit_screen.csv"))
    corr = float(tf["travel_score"].corr(tf["self_sim"]))
    out.append(Claim("§6.6", "travel score corr vs self_sim", 0.06,
                     corr, TOL["round2"], "travel_fit_screen.py",
                     "travel_fit_screen.csv"))
    y = (tf["self_sim"] > tf["self_sim"].median()).astype(int).to_numpy()
    s = tf["travel_score"].to_numpy(float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    ranks = pd.Series(s).rank(method="average").to_numpy()
    U = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2
    auc = float(U / (n_pos * n_neg))
    out.append(Claim("§6.6", "travel score rank AUC", 0.61, auc, TOL["round2"],
                     "travel_fit_screen.py", "travel_fit_screen.csv"))

    # Destination-fit null among shaper movers (resid taxonomy cut)
    u_med = tf["origin_usage"].median()
    conn = tf[(tf["origin_usage"] < u_med) & (tf["origin_influence_resid"] > 0)].copy()
    fit_delta = float(conn["fit_score"].corr(conn["delta"]))
    fit_port = float(conn["fit_score"].corr((conn["delta"] > 0).astype(float)))
    out.append(Claim("§6.6", "fit vs delta corr (shapers)", -0.09,
                     fit_delta, TOL["round2"], "travel_fit_screen.py",
                     "travel_fit_screen.csv"))
    out.append(Claim("§6.6", "fit vs portable corr (shapers)", -0.39,
                     fit_port, TOL["round2"], "travel_fit_screen.py",
                     "travel_fit_screen.csv"))

    cut = pd.read_csv(OUT / "cut_drive_travel_screen.csv").iloc[0]
    out.append(Claim("§6.6", "cut-drive shaper n", 24,
                     int(cut["n_shapers"]), TOL["int"],
                     "cut_drive_travel_screen.py", "cut_drive_travel_screen.csv"))
    out.append(Claim("§6.6", "cut share Spearman rho", 0.32,
                     float(cut["cut_rho"]), TOL["round2"],
                     "cut_drive_travel_screen.py", "cut_drive_travel_screen.csv"))
    out.append(Claim("§6.6", "cut share rank AUC", 0.71,
                     float(cut["cut_auc"]), TOL["round2"],
                     "cut_drive_travel_screen.py", "cut_drive_travel_screen.csv"))
    out.append(Claim("§6.6", "cut-minus-drive Spearman rho", 0.38,
                     float(cut["cut_minus_drive_rho"]), TOL["round2"],
                     "cut_drive_travel_screen.py", "cut_drive_travel_screen.csv"))
    out.append(Claim("§6.6", "high-cut pct portable", 75.0,
                     float(cut["high_cut_pct_portable"]), TOL["round1"],
                     "cut_drive_travel_screen.py", "cut_drive_travel_screen.csv"))
    out.append(Claim("§6.6", "low-cut pct portable", 37.5,
                     float(cut["low_cut_pct_portable"]), TOL["round1"],
                     "cut_drive_travel_screen.py", "cut_drive_travel_screen.csv"))

    bat = pd.read_csv(OUT / "destination_battery_connectors.csv")
    q = _bh_qvalues(bat["p"].to_numpy(float))
    n_surv = int((q < 0.05).sum())
    out.append(Claim("§6.6", "destination battery n features", 63,
                     len(bat), TOL["int"],
                     "(frozen)", "destination_battery_connectors.csv"))
    out.append(Claim("§6.6", "destination FDR survivors q<0.05", 0,
                     n_surv, TOL["int"],
                     "(frozen)", "destination_battery_connectors.csv"))
    return out


def claims_appendix():
    app = pd.read_csv(OUT / "connector_appendix_2526.csv")
    return [Claim("Appendix G", "2025-26 shaper roster n", 41,
                  len(app), TOL["int"],
                  "(frozen)", "connector_appendix_2526.csv")]


# ---------------------------------------------------------------------------
# sheet + main
# ---------------------------------------------------------------------------
def write_sheet(claims):
    lines = [
        "# PAPER_NUMBERS_SHEET - Structural Influence",
        "",
        f"_Generated: {date.today().isoformat()} by `_paper_numbers_audit.py`_",
        "_Source: `paper.tex`. Values recomputed from `output/` (no Neo4j)._",
        "",
        "Run `python _paper_numbers_audit.py --check` (must exit 0).",
        "",
        "| Status | Section | Claim | Paper | Actual | Tol | Script | Artifact | Note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    n_fail = 0
    for c in claims:
        st = "PASS" if c.ok else "FAIL"
        if not c.ok:
            n_fail += 1

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        note = (c.note or "").replace("|", "/")
        lines.append(
            f"| {st} | {c.section} | {c.claim} | {fmt(c.paper)} | "
            f"{fmt(c.actual)} | {c.tol} | `{c.script}` | `{c.artifact}` | {note} |"
        )
    lines.append("")
    lines.append(f"**Summary:** {len(claims) - n_fail}/{len(claims)} PASS, "
                 f"{n_fail} FAIL.")
    lines.append("")
    SHEET.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify only; exit 1 on FAIL")
    args = ap.parse_args()

    print("Recomputing paper numbers from output/ ...")
    claims = []
    for fn in (
        claims_data,
        claims_taxonomy,
        claims_exemplars,
        claims_ortho,
        claims_behavior,
        claims_efficiency,
        claims_connector_tiers,
        claims_identity,
        claims_portability,
        claims_travelfit,
        claims_appendix,
    ):
        print(f"  {fn.__name__} ...", flush=True)
        claims.extend(fn())

    n_fail = write_sheet(claims)
    n_pass = len(claims) - n_fail
    print()
    print("=" * 64)
    print(f"PAPER NUMBERS AUDIT: {n_pass}/{len(claims)} PASS, {n_fail} FAIL")
    print(f"Wrote {SHEET}")
    print("=" * 64)
    if n_fail:
        print("\nFAILING claims:")
        for c in claims:
            if not c.ok:
                print(f"  [{c.section}] {c.claim}: paper={c.paper} "
                      f"actual={c.actual}  ({c.note})")
    if args.check and n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
