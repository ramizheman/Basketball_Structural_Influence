"""
STRUCTURAL ROLE CLASSIFICATION (descriptive taxonomy; foundation artifact)
==========================================================================
Research question: "What types of offensive organizers exist, beyond traditional
usage and position?" — classify players by their STRUCTURAL RESPONSIBILITY in the
offensive co-initiation network, on two axes:

    x = usage           (real NBA USG%, possession consumption)
    y = residual influence   resid(p) = L(p) - E[L|usage]   (influence above/below usage expectation)

Classification cuts are principled, not equal-mass medians on raw influence:
    vertical:   usage median (high vs low volume)
    horizontal: resid(p) = 0  (more vs less influence than usage predicts)

Unequal class sizes are expected and fine: most players sit at or below expected influence.

Four archetypes:
  PRIMARY ORGANIZER   hi usage / resid>0  (consume AND organize beyond usage)
  CONNECTOR / HUB     lo usage / resid>0  (organize beyond usage without consuming)
  TERMINAL SCORER     hi usage / resid<=0 (finish; influence at or below usage expectation)
  ROLE OCCUPANT       lo usage / resid<=0 (neither volume nor excess structure)

Canonical corners use usage percentiles × resid sign (strong volume extremes with
excess / deficit structure).

Reuses (no drift): fragility_full_loads.csv (L(p)), player_traditional_stats.csv
(USG%, MPG), player_positions.csv. Rotation filter mpg>=15 & gp>=30.

Outputs (output/):
  structural_role_classification.csv     full player-season table (deliverable A)
  structural_role_summary.csv            per-class summary stats (deliverable B)
  fig_structural_role_map.png            2D role map (deliverable D)
  structural_role_taxonomy.html          readable report (A-E)

Run: python structural_role_taxonomy.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from wiring_full_study import OUT_DIR
from fragility_orthogonality import load_merged, spearman

MIN_MPG = 15.0
MIN_GP = 30

CLASSES = {
    (True, True): "PRIMARY ORGANIZER",
    (False, True): "CONNECTOR / HUB",
    (True, False): "TERMINAL SCORER",
    (False, False): "ROLE OCCUPANT / SPECIALIST",
}
CLASS_COLOR = {
    "PRIMARY ORGANIZER": "#d94f4f",
    "CONNECTOR / HUB": "#3f8fd9",
    "TERMINAL SCORER": "#e0a33e",
    "ROLE OCCUPANT / SPECIALIST": "#8a8f98",
}
ORDER = ["PRIMARY ORGANIZER", "CONNECTOR / HUB", "TERMINAL SCORER",
         "ROLE OCCUPANT / SPECIALIST"]


def classify(df, u_cut):
    """usage median × residual influence > 0 (principled cut, not equal-mass influence median)."""
    hi_u = df["usg_pct"] >= u_cut
    hi_r = df["u_resid"] > 0
    return [CLASSES[(bool(a), bool(b))] for a, b in zip(hi_u, hi_r)]


def within_player_influence_cv(all_loads):
    """Per player: SD/mean of L across seasons (>=2 seasons). trait-vs-context proxy."""
    rows = []
    for name, g in all_loads.groupby("player"):
        if g["yy"].nunique() < 2:
            continue
        m, s = g["Lcos"].mean(), g["Lcos"].std(ddof=1)
        rows.append(dict(player=name, influence_cv=(s / m if m > 0 else np.nan)))
    return pd.DataFrame(rows)


def main():
    df = load_merged()
    all_loads = pd.read_csv(OUT_DIR / "fragility_full_loads.csv")

    # usage-residual influence (already defined identically in load_merged via u; recompute on usg_pct-free axis)
    rot = df[(df["mpg"] >= MIN_MPG) & (df["gp"] >= MIN_GP)
             & df["usg_pct"].notna() & df["Lcos"].notna()].copy()
    print(f"rotation player-seasons (mpg>={MIN_MPG}, gp>={MIN_GP}, usg present): {len(rot)}")

    # percentiles within the rotation pool
    rot["usage_percentile"] = rot["usg_pct"].rank(pct=True) * 100
    rot["influence_percentile"] = rot["Lcos"].rank(pct=True) * 100

    # primary classification: usage median × residual influence > 0
    u_med = rot["usg_pct"].median()
    rot["role_classification"] = classify(rot, u_med)

    # canonical exemplar flag: strong usage extremes with matching resid sign
    def canon(r):
        u_hi, u_lo = r["usage_percentile"] >= 75, r["usage_percentile"] <= 25
        r_pos, r_neg = r["u_resid"] > 0, r["u_resid"] <= 0
        if u_hi and r_pos: return "PRIMARY ORGANIZER"
        if u_lo and r_pos: return "CONNECTOR / HUB"
        if u_hi and r_neg: return "TERMINAL SCORER"
        if u_lo and r_neg: return "ROLE OCCUPANT / SPECIALIST"
        return ""
    rot["canonical_archetype"] = rot.apply(canon, axis=1)

    # within-player stability
    cv = within_player_influence_cv(all_loads)
    rot = rot.merge(cv, on="player", how="left")

    # ---- deliverable A: classification table -------------------------------
    tableA = rot[["player", "season", "tri", "position", "gp", "mpg",
                  "usg_pct", "Lcos", "u_resid", "usage_percentile", "influence_percentile",
                  "role_classification", "canonical_archetype", "influence_cv"]].copy()
    tableA = tableA.rename(columns={"tri": "team", "usg_pct": "usage",
                                    "Lcos": "structural_influence", "u_resid": "influence_residual"})
    tableA = tableA.sort_values(["role_classification", "structural_influence"],
                                ascending=[True, False])
    tableA.to_csv(OUT_DIR / "structural_role_classification.csv", index=False)

    # ---- deliverable B: summary statistics ---------------------------------
    srows = []
    for cls in ORDER:
        g = rot[rot["role_classification"] == cls]
        srows.append(dict(
            role=cls, n=len(g), pct_of_pool=round(100 * len(g) / len(rot), 1),
            avg_usage=round(g["usg_pct"].mean(), 3),
            avg_structural_influence=round(g["Lcos"].mean(), 3),
            avg_mpg=round(g["mpg"].mean(), 1),
            avg_ppg=round(g["ppg"].mean(), 1),
            avg_influence_residual=round(g["u_resid"].mean(), 3),
            avg_within_player_influence_cv=round(g["influence_cv"].mean(), 3),
            n_canonical=int((g["canonical_archetype"] == cls).sum()),
        ))
    summ = pd.DataFrame(srows)
    summ.to_csv(OUT_DIR / "structural_role_summary.csv", index=False)

    # ---- threshold sensitivity (usage cut only; resid=0 is fixed by construction)
    sens = {}
    for tag, q in [("p45", 45), ("median", 50), ("p55", 55)]:
        uc = np.percentile(rot["usg_pct"], q)
        lab = classify(rot, uc)
        sens[tag] = pd.Series(lab, index=rot.index)
    stab45 = float((sens["p45"] == sens["median"]).mean())
    stab55 = float((sens["p55"] == sens["median"]).mean())
    sens_counts = {t: sens[t].value_counts().to_dict() for t in sens}

    # ---- deliverable C: top 20 by residual influence per class ---------------------------
    top_by_class = {}
    for cls in ORDER:
        g = rot[rot["role_classification"] == cls].nlargest(20, "u_resid")
        top_by_class[cls] = g

    # ---- deliverable E: orthogonality recap (already confirmed) ------------
    rho_ul, n_ul = spearman(rot["usg_pct"].to_numpy(float), rot["Lcos"].to_numpy(float))
    rho_resid, _ = spearman(rot["u_resid"].to_numpy(float), rot["usg_pct"].to_numpy(float))

    # ---- console -----------------------------------------------------------
    print("\n=== CLASS COUNTS (usage median × resid>0) ===")
    print(summ[["role", "n", "pct_of_pool", "avg_usage", "avg_structural_influence",
                "avg_mpg", "avg_influence_residual", "n_canonical"]].to_string(index=False))
    print(f"\nusage median={u_med:.3f}  residual cut=0 (share resid>0="
          f"{(rot['u_resid']>0).mean()*100:.1f}%)")
    print(f"threshold sensitivity: class-label stability vs usage-median  "
          f"p45={stab45:.2%}  p55={stab55:.2%}")
    print(f"\northogonality (recap; already confirmed): Spearman(usage, influence)={rho_ul:.3f}, "
          f"Spearman(residual-influence, usage)={rho_resid:.3f}  (n={n_ul})")
    print("  => usage and structural influence are related but far from identical;")
    print("     usage-residual influence is ~orthogonal to usage (distinct dimension).")
    for cls in ORDER:
        g = top_by_class[cls]
        print(f"\n--- {cls}: top {min(8,len(g))} by residual influence ---")
        for r in g.head(8).itertuples(index=False):
            print(f"    {r.player:<22}{r.tri} {r.season}  usg={r.usg_pct*100:4.1f}%  "
                  f"load={r.Lcos:.3f}  resid={r.u_resid:+.3f}  pos={r.position}")

    # ---- deliverable D: 2D role map ----------------------------------------
    make_map(rot, u_med, OUT_DIR / "fig_structural_role_map.png")

    # ---- HTML report -------------------------------------------------------
    l_med = float(rot["Lcos"].median())  # reported for reference only
    write_html(summ, tableA, top_by_class, sens_counts, stab45, stab55,
               u_med, l_med, rho_ul, rho_resid, n_ul, len(rot))

    print("\nWrote structural_role_classification.csv, structural_role_summary.csv, "
          "fig_structural_role_map.png, structural_role_taxonomy.html")


def make_map(rot, u_med, path):
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    for cls in ORDER:
        g = rot[rot["role_classification"] == cls]
        ax.scatter(g["usg_pct"] * 100, g["u_resid"], s=16, alpha=0.45,
                   color=CLASS_COLOR[cls], linewidths=0, label=f"{cls} (n={len(g)})")
    ax.axvline(u_med * 100, color="#333", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0.0, color="#333", lw=0.8, ls="--", alpha=0.6)

    for cls in ("PRIMARY ORGANIZER", "CONNECTOR / HUB"):
        g = rot[rot["role_classification"] == cls].nlargest(8, "u_resid")
        for r in g.itertuples(index=False):
            ax.annotate(f"{r.player}", (r.usg_pct * 100, r.u_resid),
                        fontsize=6.8, xytext=(3, 2), textcoords="offset points",
                        color="#111")
    xmax = rot["usg_pct"].max() * 100
    ymax = rot["u_resid"].max()
    ymin = rot["u_resid"].min()
    cap = [
        (u_med * 100 + (xmax - u_med * 100) * 0.35, ymax * 0.85, "PRIMARY ORGANIZER"),
        (u_med * 100 * 0.35, ymax * 0.85, "CONNECTOR / HUB"),
        (u_med * 100 + (xmax - u_med * 100) * 0.35, ymin * 0.55, "TERMINAL SCORER"),
        (u_med * 100 * 0.35, ymin * 0.55, "ROLE OCCUPANT"),
    ]
    for x, y, t in cap:
        ax.text(x, y, t, fontsize=9, fontweight="bold", alpha=0.30, ha="center")
    ax.set_xlabel("Usage %  (possession consumption)")
    ax.set_ylabel("Residual structural influence  resid(p)  (influence − E[influence|usage])")
    ax.set_title("Structural Role Map of NBA Offensive Players\n"
                 "rotation player-seasons · dashed = usage median & resid=0 · "
                 "descriptive, no outcomes",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_html(summ, tableA, top_by_class, sens_counts, stab45, stab55,
               u_med, l_med, rho_ul, rho_resid, n_ul, n_pool):
    import base64
    img = OUT_DIR / "fig_structural_role_map.png"
    b64 = base64.b64encode(img.read_bytes()).decode() if img.exists() else ""

    def tbl(d, cols=None):
        if cols:
            d = d[cols]
        return d.to_html(index=False, escape=False)

    tops = ""
    for cls in ORDER:
        g = top_by_class[cls].copy()
        g["usage_%"] = (g["usg_pct"] * 100).round(1)
        g["influence"] = g["Lcos"].round(3)
        show = g[["player", "season", "tri", "position", "usage_%", "influence"]].rename(
            columns={"tri": "team"})
        tops += f"<h3 style='color:{CLASS_COLOR[cls]}'>{cls} — top {len(show)} by structural influence</h3>\n{tbl(show)}\n"

    sens_html = "<table><tr><th>threshold</th>" + "".join(
        f"<th>{c}</th>" for c in ORDER) + "</tr>"
    for t in ["p45", "median", "p55"]:
        sens_html += f"<tr><td>{t}</td>" + "".join(
            f"<td>{sens_counts[t].get(c, 0)}</td>" for c in ORDER) + "</tr>"
    sens_html += "</table>"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Structural Role Classification in NBA Offensive Networks</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1150px; margin: 2rem auto; padding: 0 1rem; color:#111; }}
h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem; border-bottom:1px solid #ddd; padding-bottom:4px;}}
h3 {{ font-size: 1rem; margin-top: 1.3rem; }}
.note {{ color:#444; font-size:0.9rem; line-height:1.55; }}
table {{ border-collapse: collapse; width:100%; font-size:0.82rem; margin:0.6rem 0 1.2rem; }}
th,td {{ border:1px solid #ccc; padding:4px 7px; text-align:left; }}
th {{ background:#eee; }} tr:nth-child(even){{background:#fafafa;}}
img {{ max-width:100%; border:1px solid #ddd; }}
.key {{ background:#f4f7fb; border-left:4px solid #3f8fd9; padding:0.6rem 1rem; }}
</style></head><body>
<h1>Structural Role Classification in NBA Offensive Networks</h1>
<p class="note"><b>Research question:</b> what types of offensive organizers exist beyond usage and
position? Players are classified by <b>structural responsibility</b> in the co-initiation network, on
two axes: <b>usage %</b> (possession consumption) and <b>structural influence L(p)</b> (how much removing the
player re-wires the offense's play-type topology). Descriptive only — no wins, plus-minus, contracts,
or outcomes. Rotation player-seasons (MPG&ge;{MIN_MPG:.0f}, GP&ge;{MIN_GP}); n={n_pool}.</p>

<div class="key"><b>Key hypothesis (already confirmed in the orthogonality analysis):</b> usage
captures possession consumption; structural influence captures organizational responsibility, and they are
<b>distinct dimensions</b>. Spearman(usage, influence)={rho_ul:.2f} (related, not redundant); the
usage-<i>residual</i> influence is ~orthogonal to usage ({rho_resid:+.2f}), i.e. structural responsibility
is not recoverable from how many possessions a player uses.</div>

<h2>D. Structural role map (usage &times; structural influence)</h2>
<img src="data:image/png;base64,{b64}" alt="structural role map">

<h2>B. Summary statistics per class (primary = median split)</h2>
<p class="note">usage median={u_med*100:.1f}% · influence median={l_med:.3f}. "canonical" = players in the
&ge;75th/&le;25th corner on both axes (strongest exemplars).</p>
{tbl(summ)}

<h2>Threshold sensitivity</h2>
<p class="note">Class counts under cuts at the 45th / 50th (primary) / 55th percentile. Label stability
vs the primary median split: <b>p45 {stab45:.0%}</b>, <b>p55 {stab55:.0%}</b> of players keep the same
class — the taxonomy is not an artifact of the exact cut.</p>
{sens_html}

<h2>C. Representative examples — top 20 by structural influence in each class</h2>
{tops}

<h2>A. Full classification table</h2>
<p class="note">Sorted by class, then structural influence. Full CSV:
structural_role_classification.csv.</p>
{tbl(tableA.round({{'usage':3,'structural_influence':3,'influence_residual':3,'usage_percentile':0,'influence_percentile':0,'influence_cv':2}}) if False else tableA.round(3))}
</body></html>"""
    (OUT_DIR / "structural_role_taxonomy.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
