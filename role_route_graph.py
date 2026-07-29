"""
ROLE-ROUTE GRAPH  (the graph object made explicit; descriptive)
===============================================================
Per possession we observe the INITIATING player's role and the play type, giving a
real 2-node route:  role -> play_type  (e.g. organizer-PnR, connector-handoff).
NOTE: intra-possession passing chains (organizer->connector->cutter) are NOT in the
data; the honest graph object is this bipartite role->play_type routing network.

For each on-court role composition we build the weighted bipartite route graph and
measure:
  * active routes    : # distinct (role,play_type) routes with share >= 1%
  * route entropy     : Shannon entropy over the route distribution
Contrast connector-present vs connector-absent, differenced WITHIN team-season.

MECHANICAL CAVEAT: adding ANY new initiator role trivially adds its own routes, so
raw route counts rise by construction. The clean measure is NON-FOCAL route entropy
(diversity of the OTHER roles' routes) — does a connector reorganize everyone else's
routing, not just add its own. Both are reported; the non-focal one is the headline.

Outputs (output/):
  role_route_graph.csv, fig_role_route_graph.png, role_route_graph.html

Run: python role_route_graph.py
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

from wiring_gate import PLAY_TYPES_8, OUT_DIR
from role_composition_playmix import pull_possessions, build_class_map, ORDER

MIN_POSS_STRATUM = 200
ROUTE_ACTIVE_MIN = 0.01


def entropy(counts):
    p = np.asarray(counts, float)
    tot = p.sum()
    if tot <= 0:
        return 0.0
    p = p[p > 0] / tot
    return float(-(p * np.log(p)).sum())


def main():
    df = pull_possessions()
    cmap, _, _ = build_class_map()
    team = df["team"].to_numpy(); yy = df["yy"].to_numpy()

    df["init_role"] = [cmap.get((team[i], int(yy[i]), df["initiator"].iat[i]), "?")
                       for i in range(len(df))]
    plcols = [f"pl{i}" for i in range(5)]
    comp = {k: np.zeros(len(df), np.int16) for k in ORDER}
    for c in plcols:
        names = df[c].to_numpy()
        arr = np.array([cmap.get((team[i], int(yy[i]), names[i]), "?") for i in range(len(df))],
                       dtype=object)
        for k in ORDER:
            comp[k] += (arr == k)
    for k in ORDER:
        df[f"n_{k}"] = comp[k]
    df = df[df["init_role"] != "?"].copy()
    df["_ts"] = df["team"] + "_" + df["yy"].astype(str)
    print(f"possessions with classified initiator: {len(df):,}")

    # route id = (role, play_type)
    routes = [(r, p) for r in ORDER for p in PLAY_TYPES_8]
    ridx = {rp: i for i, rp in enumerate(routes)}
    rvec = np.array([ridx[(r, p)] for r, p in zip(df["init_role"], df["ptype"])])
    df["_route"] = rvec

    def route_counts(idx):
        return np.bincount(rvec[idx], minlength=len(routes)).astype(float)

    def nonfocal_entropy(idx, focal):
        """entropy over routes NOT initiated by `focal` role."""
        c = route_counts(idx)
        for (r, p), i in ridx.items():
            if r == focal:
                c[i] = 0
        return entropy(c)

    # within-team-season connector present vs absent
    print("\n=== within-team-season: CONNECTOR present(>=1) vs absent(0) ===")
    rows = []
    for focal, nk_col in [("CONNECTOR", "n_CONNECTOR"), ("ORGANIZER", "n_ORGANIZER")]:
        nk = df[nk_col].to_numpy()
        dE_all, dE_nf, dCnt, nts = [], [], [], 0
        for ts, idx in df.groupby("_ts").indices.items():
            idx = np.asarray(idx)
            wi = idx[nk[idx] >= 1]; wo = idx[nk[idx] == 0]
            if len(wi) < MIN_POSS_STRATUM or len(wo) < MIN_POSS_STRATUM:
                continue
            cw, co = route_counts(wi), route_counts(wo)
            e_all = entropy(cw) - entropy(co)
            e_nf = nonfocal_entropy(wi, focal) - nonfocal_entropy(wo, focal)
            n_w = int(((cw / cw.sum()) >= ROUTE_ACTIVE_MIN).sum())
            n_o = int(((co / co.sum()) >= ROUTE_ACTIVE_MIN).sum())
            dE_all.append(e_all); dE_nf.append(e_nf); dCnt.append(n_w - n_o); nts += 1
        print(f"\n{focal} present vs absent  (team-seasons n={nts})")
        print(f"  Δ route entropy (ALL routes, incl. mechanical): {np.mean(dE_all):+.4f}")
        print(f"  Δ route entropy (NON-focal, clean headline)   : {np.mean(dE_nf):+.4f}")
        print(f"  Δ active-route count (>=1% share)             : {np.mean(dCnt):+.2f}")
        rows.append(dict(focal=focal, n_ts=nts,
                         d_entropy_all=round(float(np.mean(dE_all)), 4),
                         d_entropy_nonfocal=round(float(np.mean(dE_nf)), 4),
                         d_active_routes=round(float(np.mean(dCnt)), 2)))
    pd.DataFrame(rows).to_csv(OUT_DIR / "role_route_graph.csv", index=False)

    make_figure(df, rvec, ridx, routes)
    print("\nWrote role_route_graph.csv, fig_role_route_graph.png, role_route_graph.html")
    write_html(rows)


def make_figure(df, rvec, ridx, routes):
    """Bipartite role->play_type graph, connector-absent vs connector-present (pooled)."""
    nc = df["n_CONNECTOR"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=150)
    for ax, mask, title in [
        (axes[0], nc == 0, "CONNECTOR ABSENT (0 on floor)"),
        (axes[1], nc >= 1, "CONNECTOR PRESENT (>=1 on floor)"),
    ]:
        idx = np.where(mask)[0]
        counts = np.bincount(rvec[idx], minlength=len(routes)).astype(float)
        share = counts / counts.sum()
        ry = {r: i for i, r in enumerate(ORDER[::-1])}
        py = {p: i for i, p in enumerate(PLAY_TYPES_8[::-1])}
        rcol = {"CONNECTOR": "#3f8fd9", "ORGANIZER": "#d94f4f",
                "TERMINAL": "#e0a33e", "OCCUPANT": "#8a8f98"}
        n_active = 0
        for (r, p), i in ridx.items():
            s = share[i]
            if s < ROUTE_ACTIVE_MIN:
                continue
            n_active += 1
            ax.plot([0, 1], [ry[r] / (len(ORDER) - 1) * 7, py[p]], color=rcol[r],
                    lw=s * 60, alpha=0.55, solid_capstyle="round")
        for r, yv in ry.items():
            ax.text(-0.03, yv / (len(ORDER) - 1) * 7, r, ha="right", va="center",
                    fontsize=9, fontweight="bold", color=rcol[r])
        for p, yv in py.items():
            ax.text(1.03, yv, p, ha="left", va="center", fontsize=8)
        ax.set_xlim(-0.35, 1.4); ax.set_ylim(-0.6, 7.6)
        ax.set_title(f"{title}\nactive routes (>=1%): {n_active}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Role -> Play-Type Route Graph (initiating role -> play run)\n"
                 "edge width proportional to share of possessions · descriptive", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "fig_role_route_graph.png")
    plt.close(fig)


def write_html(rows):
    import base64
    img = OUT_DIR / "fig_role_route_graph.png"
    b64 = base64.b64encode(img.read_bytes()).decode() if img.exists() else ""
    tr = "".join(
        f"<tr><td>{r['focal']}</td><td>{r['n_ts']}</td><td>{r['d_entropy_all']:+.4f}</td>"
        f"<td><b>{r['d_entropy_nonfocal']:+.4f}</b></td><td>{r['d_active_routes']:+.2f}</td></tr>"
        for r in rows)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Role-Route Graph</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#111}}
h1{{font-size:1.5rem}} h2{{font-size:1.1rem;margin-top:1.6rem;border-bottom:1px solid #ddd}}
.note{{color:#444;font-size:0.9rem;line-height:1.55}}
.warn{{background:#fff6e5;border-left:4px solid #e0a33e;padding:0.6rem 1rem;font-size:0.9rem}}
table{{border-collapse:collapse;font-size:0.88rem;margin:0.6rem 0}} th,td{{border:1px solid #ccc;padding:5px 9px}}
th{{background:#eee}} img{{max-width:100%;border:1px solid #ddd}}</style></head><body>
<h1>Role &rarr; Play-Type Route Graph</h1>
<p class="note">The graph object made explicit: each possession's <b>initiating role</b> &rarr;
<b>play type</b> is a route. Edge width = share of possessions. We contrast the routing network when
a connector is absent vs present on the floor.</p>
<div class="warn"><b>Mechanical caveat:</b> adding any initiator role trivially adds its own routes, so
raw route counts/entropy rise by construction. The clean measure is <b>non-focal route entropy</b>
(diversity of the OTHER roles' routes) &mdash; does a connector reorganize everyone else's routing.</div>
<h2>Figure</h2>
<img src="data:image/png;base64,{b64}" alt="role route graph">
<h2>Within-team-season route diversity change (present &minus; absent)</h2>
<table><tr><th>focal role</th><th>team-seasons</th><th>Δ entropy (all, mechanical)</th>
<th>Δ entropy (non-focal, clean)</th><th>Δ active routes</th></tr>{tr}</table>
<p class="note">A positive <b>non-focal</b> Δ means the presence of that role makes the OTHER roles'
routing more diverse &mdash; genuine reorganization, not mechanical addition.</p>
</body></html>"""
    (OUT_DIR / "role_route_graph.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
