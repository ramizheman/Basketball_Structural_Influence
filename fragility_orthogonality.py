"""
ORTHOGONALITY, QUADRANTS & WITHIN-PLAYER STABILITY (exploratory, §8)
====================================================================

The reviewer-proofing analysis: is structural influence just another impact metric?

(1) Orthogonality — Spearman of structural influence (and usage-residual influence)
    against usage, touches, front-court touches, points, assists, assist%,
    and on-court net rating. Structural influence measuring a DIFFERENT axis =>
    moderate/low correlation, and specifically identifying cases the others
    miss (the off-diagonal).

(2) Quadrant matrix — usage (median split) x structural influence (median split),
    among rotation players. Populated quadrants = it measures something
    distinct from production. Names the four cells.

(3) Within-player stability — is structural identity a player trait or a team-
    context trait? Within-player SD across seasons vs the between-player SD;
    name the stable hubs (SGA) vs the volatile ones (Gobert).

All stats are official NBA data (fetch_traditional_stats.py); structural influence
is our graph metric. Merged by (tri, yy, abbrev).

Writes a console report + output/fragility_orthogonality.csv and a small
figure output/fig4_load_vs_metrics.png.

Run: python fragility_orthogonality.py
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

LOADS = OUT_DIR / "fragility_full_loads.csv"
STATS = OUT_DIR / "player_traditional_stats.csv"
POS = OUT_DIR / "player_positions.csv"

MIN_MPG = 15.0   # rotation-player floor for the quadrant table
MIN_GP = 30


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        return np.nan, 0
    rx = pd.Series(x).rank().to_numpy().copy()
    ry = pd.Series(y).rank().to_numpy().copy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return (float((rx * ry).sum() / d) if d > 0 else np.nan), len(x)


def load_merged(season_yy=(22, 23, 24)):
    """Canonical residual-influence construction, matching paper.tex Eq. 5
    (sec:resid): quadratic fit of Lcos ~ u (graph-native initiation share)
    across all classified players (no mpg/gp restriction on the fit sample;
    season restricted to the pre-registered 2022–25 sample by default).
    Shared by role_composition_playmix.build_class_map().
    """
    loads = pd.read_csv(LOADS)
    loads["abbrev"] = loads["player"]   # graph name already 'F. Last' format
    stats = pd.read_csv(STATS)
    pos = pd.read_csv(POS)
    pos_map = {(r.tri, int(r.yy), r.abbrev): r.position for r in pos.itertuples(index=False)}

    df = loads.merge(stats, on=["tri", "yy", "abbrev"], how="left",
                     suffixes=("", "_nba"))
    df["position"] = [pos_map.get((r.tri, int(r.yy), r.abbrev), "?")
                      for r in df.itertuples(index=False)]

    if season_yy is not None:
        df = df[df["yy"].isin(season_yy)].copy()

    u = df["u"].to_numpy(float)
    l = df["Lcos"].to_numpy(float)
    ok = np.isfinite(u) & np.isfinite(l)
    X = np.column_stack([np.ones(ok.sum()), u[ok], u[ok] ** 2])
    beta, *_ = np.linalg.lstsq(X, l[ok], rcond=None)
    resid = np.full(len(df), np.nan)
    resid[ok] = l[ok] - X @ beta
    df["u_resid"] = resid
    return df


def orthogonality(df):
    metrics = [("usg_pct", "Usage %"), ("touches", "Touches/g"),
               ("front_ct_touches", "Frontcourt touches/g"),
               ("ppg", "Points/g"), ("apg", "Assists/g"),
               ("ast_pct", "Assist %"), ("net_rating", "On-court net rating")]
    print("\n" + "=" * 72)
    print("(1) ORTHOGONALITY — Spearman vs conventional metrics")
    print("=" * 72)
    print(f"{'Metric':<24}{'rho(influence)':>11}{'rho(resid)':>12}{'n':>7}")
    out = []
    load = df["Lcos"].to_numpy()
    resid = df["u_resid"].to_numpy()
    for col, label in metrics:
        v = df[col].to_numpy(dtype=float)
        r_load, n = spearman(load, v)
        r_res, _ = spearman(resid, v)
        out.append(dict(metric=label, rho_load=r_load, rho_resid=r_res, n=n))
        print(f"{label:<24}{r_load:>11.3f}{r_res:>12.3f}{n:>7}")
    print("\n  Interpretation: high influence correlates moderately with usage/touches")
    print("  (bigger role => more displacement, expected). The RESIDUAL column is")
    print("  the key: near-zero vs production => structural influence is a different axis.")
    return pd.DataFrame(out)


def quadrants(df):
    rot = df[(df["mpg"] >= MIN_MPG) & (df["gp"] >= MIN_GP)].copy()
    u_med = rot["usg_pct"].median()
    l_med = rot["Lcos"].median()
    rot["hi_usage"] = rot["usg_pct"] >= u_med
    rot["hi_load"] = rot["Lcos"] >= l_med
    print("\n" + "=" * 72)
    print(f"(2) QUADRANT MATRIX  (rotation players: mpg>={MIN_MPG}, gp>={MIN_GP}; "
          f"n={len(rot)})")
    print(f"    usage median={u_med:.3f}   influence median={l_med:.3f}")
    print("=" * 72)

    def cell(hi_u, hi_l, sort_col, n=8):
        c = rot[(rot["hi_usage"] == hi_u) & (rot["hi_load"] == hi_l)]
        return c.nlargest(n, sort_col)

    quads = [
        ("HIGH usage / HIGH influence  = Offensive superstructures", True, True, "Lcos"),
        ("LOW usage  / HIGH influence  = Hidden organizers", False, True, "Lcos"),
        ("HIGH usage / LOW influence   = Volume scorers", True, False, "usg_pct"),
        ("LOW usage  / LOW influence   = Replaceable pieces", False, False, "usg_pct"),
    ]
    for title, hu, hl, sc in quads:
        c = cell(hu, hl, sc)
        cnt = int(((rot["hi_usage"] == hu) & (rot["hi_load"] == hl)).sum())
        print(f"\n  {title}  (n={cnt})")
        for r in c.itertuples(index=False):
            print(f"    {r.player:<22}{r.tri} {r.season}  "
                  f"usg={r.usg_pct*100:4.1f}%  influence={r.Lcos:.3f}  "
                  f"pts={r.ppg:4.1f}  pos={r.position}")
    return rot


def stability(df):
    print("\n" + "=" * 72)
    print("(3) WITHIN-PLAYER STABILITY — trait vs team-context")
    print("=" * 72)
    # players with >= 3 seasons in the dataset
    g = df.groupby("player")
    rows = []
    for name, sub in g:
        if sub["yy"].nunique() < 3:
            continue
        rows.append(dict(player=name, seasons=sub["yy"].nunique(),
                         mean_load=sub["Lcos"].mean(), sd_load=sub["Lcos"].std(ddof=1),
                         min_load=sub["Lcos"].min(), max_load=sub["Lcos"].max(),
                         mean_usage=sub["u"].mean() * 100))
    s = pd.DataFrame(rows)
    between_sd = df["Lcos"].std(ddof=1)
    print(f"  between-player SD of influence (all): {between_sd:.3f}")
    print(f"  players with >=3 seasons: {len(s)}")

    # focus on players who are ever high-influence (max >= 0.10): are they stable?
    hubs = s[s["max_load"] >= 0.10].sort_values("mean_load", ascending=False)
    print(f"\n  Structural hubs (max influence >= 0.10), by mean influence — is it a trait?")
    print(f"    {'Player':<22}{'seasons':>8}{'mean':>7}{'sd':>7}{'min':>7}{'max':>7}"
          f"{'cv':>7}  verdict")
    for r in hubs.head(20).itertuples(index=False):
        cv = r.sd_load / r.mean_load if r.mean_load > 0 else np.nan
        verdict = "STABLE trait" if cv < 0.5 else ("volatile" if cv > 0.9 else "moderate")
        print(f"    {r.player:<22}{r.seasons:>8}{r.mean_load:>7.3f}{r.sd_load:>7.3f}"
              f"{r.min_load:>7.3f}{r.max_load:>7.3f}{cv:>7.2f}  {verdict}")
    return s


def figure(df, out_path):
    metrics = [("usg_pct", "Usage %", 100), ("touches", "Touches / game", 1),
               ("ppg", "Points / game", 1), ("net_rating", "On-court net rating", 1)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    for ax, (col, label, scale) in zip(axes.ravel(), metrics):
        v = df[col].to_numpy(dtype=float) * scale
        load = df["Lcos"].to_numpy()
        m = np.isfinite(v) & np.isfinite(load)
        ax.scatter(v[m], load[m], s=12, alpha=0.28, color="#5b6b7c", linewidths=0)
        r, _ = spearman(load, df[col].to_numpy(dtype=float))
        # label a few structural hubs
        hubs = df[df["Lcos"] >= 0.20]
        ax.scatter(hubs[col] * scale, hubs["Lcos"], s=34, color="#d94f4f",
                   edgecolors="white", linewidths=0.5, zorder=5)
        for _, hr in hubs.iterrows():
            if hr["Lcos"] >= 0.30:
                ax.annotate(hr["player"], (hr[col] * scale, hr["Lcos"]),
                            fontsize=6.5, xytext=(4, 2), textcoords="offset points")
        ax.set_xlabel(label)
        ax.set_ylabel("Structural influence")
        ax.set_title(f"Influence vs {label}   (Spearman \u03c1 = {r:.2f})", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Structural Influence vs Conventional Metrics — is it a different axis?\n"
                 "red = structural hubs (load \u2265 0.20) · exploratory (§8)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nWrote {out_path}")


def main():
    df = load_merged()
    matched = df["usg_pct"].notna().mean()
    print(f"stat match coverage: {matched*100:.1f}% of {len(df)} player-seasons")
    ortho = orthogonality(df)
    quadrants(df)
    stability(df)
    figure(df, OUT_DIR / "fig4_load_vs_metrics.png")
    df.to_csv(OUT_DIR / "fragility_orthogonality.csv", index=False)
    print(f"Wrote {OUT_DIR / 'fragility_orthogonality.csv'}")


if __name__ == "__main__":
    main()
