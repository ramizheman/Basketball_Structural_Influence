"""
Generate the paper's figures from cached output/ data (no Neo4j).

  fig1_wiring.png
  fig5b_signatures.png
  fig_loo_displacement.png
  fig_taxonomy_map.png
  fig_shaper_delta_dist.png

Run: python make_figures.py
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

from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

from wiring_gate import PLAY_TYPES_8, IU, OUT_DIR, assoc_O, null_O_stats, standardize
from wiring_full_study import pull_edges
from fragility_gate import build_B_labeled

# Single output folder for all paper figures (Overleaf: upload this folder).
FIG_DIR = Path(__file__).resolve().parent / "figures"

# ---- Diverging heatmaps: two DIFFERENT hue families on purpose ---------------
# Fig 1 = ONE chart: MIN-minus-HOU wiring difference, cell by cell.
# Rockets red = more shared initiators in Houston.
# Wolves aurora green = more shared initiators in Minnesota.
# White = teams alike on that pair.
HOU_RED = "#CE1141"       # Houston Rockets primary red
MIN_GREEN = "#78BE20"     # Minnesota Timberwolves aurora green
MIN_NAVY = "#0C2340"      # Timberwolves midnight blue (bars / accents)
WIRING_DIFF_CMAP = LinearSegmentedColormap.from_list(
    "wiring_diff",
    [
        (0.00, HOU_RED),    # more shared initiators in HOU
        (0.50, "#FFFFFF"),  # teams alike on this pair
        (1.00, MIN_GREEN),  # more shared initiators in MIN
    ],
)

# Fig 5b concordance: only same-signed origin∩dest cells are nonzero.
# Navy = both teams agree removal WEAKENS that play-type pair link.
# Coral = both teams agree removal STRENGTHENS it.
# Cream = no same-sign match. Intensity = shared |mag| after unit-norm.
PLAYER_SHIFT_CMAP = LinearSegmentedColormap.from_list(
    "player_shift",
    ["#1A3352",  # navy  — shared weaken
     "#6B8AAD",
     "#F4F1EA",  # cream — no same-sign match
     "#E07A5F",
     "#C0392B"], # coral — shared strengthen
)

SIG = pd.read_parquet(OUT_DIR / "portability_signatures.parquet")
TF = pd.read_csv(OUT_DIR / "travel_fit_screen.csv")
SIG_COLS = [f"s{i}" for i in range(28)]
N_NULL = 300
RNG = np.random.default_rng(17)
PT = [p.replace("_", "\n") for p in PLAY_TYPES_8]
PT_SHORT = [p.replace("_", " ") for p in PLAY_TYPES_8]


def upper_triangle_matrix(M):
    out = np.full((8, 8), np.nan)
    for i in range(8):
        for j in range(i + 1, 8):
            out[i, j] = M[i, j]
    return out


def upper_triangle_values(M):
    return np.array([M[i, j] for i in range(8) for j in range(i + 1, 8)], float)


def relative_wiring_matrix(A):
    """Center each team's links on its own mean so +/− = stronger/weaker than team avg."""
    mu = np.nanmean(upper_triangle_values(A))
    return A - mu


def plot_pair_heatmap(ax, M, vmax, title, cmap=WIRING_DIFF_CMAP, cbar=True,
                      cbar_label="vs. team avg.", norm=None, tick_fs=9,
                      label_fs=10, title_fs=12, cbar_fs=9):
    kwargs = dict(norm=norm) if norm is not None else dict(vmin=-vmax, vmax=vmax)
    im = ax.imshow(upper_triangle_matrix(M), cmap=cmap,
                   aspect="equal", interpolation="nearest", **kwargs)
    ax.set_xticks(range(8))
    ax.set_xticklabels(PT_SHORT, fontsize=tick_fs, rotation=45, ha="right")
    ax.set_yticks(range(8))
    ax.set_yticklabels(PT_SHORT, fontsize=tick_fs)
    ax.set_xlabel("second play type in pair", fontsize=label_fs, labelpad=4)
    ax.set_ylabel("first play type in pair", fontsize=label_fs, labelpad=4)
    ax.set_title(title, fontsize=title_fs, pad=8)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=cbar_fs)
        cb.set_label(cbar_label, fontsize=cbar_fs)
    return im


def vec_to_mat(v):
    m = np.full((8, 8), np.nan)
    m[IU] = v
    m.T[IU] = v
    return m


def unit_norm_vec(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def figure_loo_displacement(
    tri="MIN", yy=23,
    high_player="R. Gobert", low_player="K. Towns",
    high_label="Shaper", low_label="Terminal Scorer",
    seed=17, n_null=1000,
):
    """What structural influence measures: leave-one-out displacement of the
    play-type association fingerprint.

    Two panels, same team-season, same 8x8 play-type ordering, same diverging
    scale, diagonal blank: A - A_{-p} for a high-influence and a low-influence player.
    This is the exact object underlying L(p) = 1 - cos(A_full, A_{-p})
    (Eq. load in the paper) -- no new construct, just its visualization.
    """
    df = pull_edges()
    sub = df[(df["team"] == tri) & (df["season_yy"] == yy) & (df["is_playoff"] == 0)]
    B, players = build_B_labeled(sub)
    rng = np.random.default_rng(seed)
    E, sd, _ = null_O_stats(B, rng, n_null)
    A_full = standardize(assoc_O(B), E, sd)
    total = float(B.sum())

    def loo(player):
        idx = players.index(player)
        u = float(B[idx].sum()) / total
        prof = B[idx].astype(float)
        prof = prof / prof.sum() if prof.sum() > 0 else prof
        Bm = np.delete(B, idx, axis=0)
        Em, sdm, _ = null_O_stats(Bm, rng, n_null)
        Am = standardize(assoc_O(Bm), Em, sdm)
        dA = A_full - Am
        av, mv = upper_triangle_values(A_full), upper_triangle_values(Am)
        Lcos = 1.0 - float(np.dot(av, mv) / (np.linalg.norm(av) * np.linalg.norm(mv)))
        return dA, Lcos, u, prof

    dA_hi, L_hi, u_hi, prof_hi = loo(high_player)
    dA_lo, L_lo, u_lo, prof_lo = loo(low_player)

    roles = pd.read_csv(OUT_DIR / "structural_role_classification.csv")

    def resid_lookup(player):
        yy_str = f"20{yy:02d}-{yy+1:02d}"
        hit = roles[(roles.player == player) & (roles.team == tri) & (roles.season == yy_str)]
        return float(hit.iloc[0]["influence_residual"]) if not hit.empty else float("nan")

    r_hi, r_lo = resid_lookup(high_player), resid_lookup(low_player)

    vmax = np.nanmax(np.abs(np.concatenate([
        upper_triangle_values(dA_hi), upper_triangle_values(dA_lo),
    ])))

    fig = plt.figure(figsize=(11.5, 6.4), dpi=180)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 5.5], width_ratios=[1, 1],
                          hspace=0.05, wspace=0.32)

    disp_cmap = LinearSegmentedColormap.from_list(
        "loo_displacement", ["#2c5f8a", "#eef2f5", "#b0392f"],
    )

    panels = [
        (high_player, dA_hi, L_hi, u_hi, r_hi, high_label, prof_hi),
        (low_player, dA_lo, L_lo, u_lo, r_lo, low_label, prof_lo),
    ]
    for col, (player, dA, Lval, uval, rval, label, prof) in enumerate(panels):
        ax_bar = fig.add_subplot(gs[0, col])
        ax_bar.bar(range(8), prof, color="#8a8a8a", width=0.7)
        ax_bar.set_xlim(-0.5, 7.5)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        ax_bar.set_title(
            f"{player} ({label})\n$u$={uval:.3f}  $L$={Lval:.3f}  $r$={rval:+.3f}",
            fontsize=10.5, pad=4,
        )

        ax = fig.add_subplot(gs[1, col])
        im = plot_pair_heatmap(
            ax, dA, vmax,
            title="", cmap=disp_cmap, cbar=(col == 1),
            cbar_label=r"$A - A_{-p}$ (signed displacement)",
            tick_fs=8.5, label_fs=9.5,
        )

    season_label = f"20{yy:02d}\u201320{yy+1:02d}"
    fig.suptitle(
        "What structural influence measures: leave-one-out displacement of the\n"
        f"play-type association fingerprint ({tri} {season_label}, same team-season)",
        fontsize=12, y=1.02,
    )
    fig.text(
        0.5, -0.02,
        "Each heatmap is $A-A_{-p}$: the signed change in the standardized play-type\n"
        "association matrix when player $p$'s initiation row is removed. Larger, more\n"
        "structured displacement corresponds to greater cosine-based structural influence $L(p)$.\n"
        "This is a leave-one-out structural perturbation, not an observed injury or causal adaptation.",
        ha="center", va="top", fontsize=8.5, color="#333",
    )
    out = FIG_DIR / "fig_loo_displacement.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}  (L_hi={L_hi:.3f} r_hi={r_hi:+.3f}; "
          f"L_lo={L_lo:.3f} r_lo={r_lo:+.3f})")


def team_assoc_and_freq(df, tri, yy):
    sub = df[(df["team"] == tri) & (df["season_yy"] == yy) & (df["is_playoff"] == 0)]
    B, _ = build_B_labeled(sub)
    E, sd, _ = null_O_stats(B, RNG, N_NULL)
    A = standardize(assoc_O(B), E, sd)
    freq = B.sum(axis=0).astype(float)
    freq = freq / freq.sum()
    return A, freq


def cos(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# (removed unused soft_wiring helper)


# ----------------------------------------------------------------- Figure 1
def figure1():
    df = pull_edges()
    cands = [("DEN", 22), ("BOS", 23), ("GSW", 23), ("OKC", 24),
             ("MIN", 24), ("SAC", 23), ("NYK", 24), ("HOU", 24)]
    data = {}
    for tri, yy in cands:
        try:
            A, f = team_assoc_and_freq(df, tri, yy)
            data[(tri, yy)] = (A, f)
        except Exception as e:
            print(f"  skip {tri}_{yy}: {e}")
    keys = list(data)
    best, bestscore = None, -1e9
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            Ai, fi = data[keys[i]]
            Aj, fj = data[keys[j]]
            score = cos(fi, fj) - cos(Ai[IU], Aj[IU])
            if score > bestscore:
                bestscore, best = score, (keys[i], keys[j])
    (k1, k2) = best
    A1, f1 = data[k1]
    A2, f2 = data[k2]
    R1, R2 = relative_wiring_matrix(A1), relative_wiring_matrix(A2)
    menu_cos = cos(f1, f2)
    wire_cos = cos(A1[IU], A2[IU])
    print(f"Fig1 pair: {k1} vs {k2}  menu cos={menu_cos:.2f}  wiring cos={wire_cos:.2f}")

    # ONE difference heatmap on top; menu bars underneath prove "same menu."
    D = R1 - R2
    dvals = upper_triangle_values(D)
    vmax = float(np.nanmax(np.abs(dvals)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig = plt.figure(figsize=(14, 11.5), dpi=160)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.8, 1.0], hspace=0.40)
    # Balanced gutters + dedicated colorbar column so the square doesn't ride right.
    top = gs[0].subgridspec(
        1, 5, width_ratios=[1.05, 1.35, 0.07, 0.32, 1.05], wspace=0.08
    )
    ax = fig.add_subplot(top[0, 1])
    cax = fig.add_subplot(top[0, 2])
    im = plot_pair_heatmap(
        ax, D, vmax,
        f"{k1[0]} '{k1[1]}  vs.  {k2[0]} '{k2[1]}  — where the wiring differs",
        cmap=WIRING_DIFF_CMAP,
        cbar=False,
        tick_fs=11, label_fs=12, title_fs=14, cbar_fs=11,
    )
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=11)
    cb.set_label(
        f"Wolves green = more shared in {k1[0]}\n"
        f"Rockets red = more shared in {k2[0]}",
        fontsize=11,
    )

    bot = gs[1].subgridspec(1, 2, wspace=0.28)
    for col, (k, f, bar_color) in enumerate([
        (k1, f1, MIN_GREEN),
        (k2, f2, HOU_RED),
    ]):
        axb = fig.add_subplot(bot[0, col])
        axb.bar(range(8), f * 100, color=bar_color)
        axb.set_xticks(range(8))
        axb.set_xticklabels(PT, fontsize=9)
        axb.set_ylabel("play-type\nshare (%)", fontsize=11)
        axb.set_title(f"{k[0]} '{k[1]}  — play-type menu", fontsize=12)
        axb.tick_params(axis="y", labelsize=10)
        axb.spines["top"].set_visible(False)
        axb.spines["right"].set_visible(False)

    fig.suptitle(
        f"Same menu (cos = {menu_cos:.2f}), different wiring (cos = {wire_cos:.2f}).\n"
        f"Top: Wolves green = more similarity in initiators for {k1[0]}; "
        f"Rockets red = more similarity in initiators for {k2[0]}; white = teams alike.\n"
        "Bottom: nearly identical play-type shares — the menu matches; the wiring does not.",
        fontsize=13,
    )
    fig.savefig(FIG_DIR / "fig1_wiring.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_wiring.png")


def _sig_row(name_sub, tri, yy):
    m = SIG[(SIG["player"].str.contains(name_sub, na=False)) &
            (SIG["tri"] == tri) & (SIG["yy"] == yy)]
    if m.empty:
        return None
    return m.iloc[0][SIG_COLS].to_numpy(float)


def concordance_signature(v_orig, v_dest):
    """One vector encoding overlap of origin and destination displacement signatures.

    Same-sign cells get a shared pop strength (geometric mean of |mags| with shared sign).
    Opposite-sign / near-zero cells stay near cream. Self-sim is still computed on the
    raw vectors separately.
    """
    o = np.asarray(v_orig, float)
    d = np.asarray(v_dest, float)
    out = np.zeros_like(o)
    for i in range(len(o)):
        if o[i] * d[i] > 0:
            out[i] = np.sign(o[i]) * np.sqrt(abs(o[i]) * abs(d[i]))
        # else leave 0 — no match, no pop
    return out


# ----------------------------------------------------------------- Figure 5b
def figure5b():
    # Three concordance heatmaps across the portability distribution (not archetypes):
    # upper-tail traveler, population-median mover, lower-tail non-traveler.
    CCOLORS = {
        "strong traveler": "#1a6b3c",
        "typical mover": "#2f5f8f",
        "weak traveler": "#999999",
    }
    cases = [
        # Selected by realized Delta rank among shaper movers in the
        # pre-registered n=294 window (enough initiations both sides).
        ("strong traveler", "D. Gafford", "Gafford",
         ("WAS", 23), ("DAL", 23), "WAS '23 → DAL '23"),
        ("typical mover", "D. Jones Jr.", "Jones Jr.",
         ("CHI", 22), ("DAL", 23), "CHI '22 → DAL '23"),
        ("weak traveler", "D. Wright", "Wright",
         ("WAS", 23), ("MIA", 23), "WAS '23 → MIA '23"),
    ]

    prepared = []
    all_abs = []
    for tag, name, sub, (t1, y1), (t2, y2), move_lbl in cases:
        v1 = _sig_row(sub, t1, y1)
        v2 = _sig_row(sub, t2, y2)
        if v1 is None or v2 is None:
            prepared.append(None)
            continue
        sc = cos(v1, v2)
        # Raw displacement magnitude scales with roster/possession volume, not
        # portability (e.g. Gafford's raw |v| is much larger than Jones Jr.'s). self_sim
        # is a cosine (scale-free) similarity, so the concordance heatmap must be
        # computed on unit-normalized vectors too, or a shared color scale
        # across players washes out real, smaller-magnitude overlap.
        c = concordance_signature(unit_norm_vec(v1), unit_norm_vec(v2))
        all_abs.extend(np.abs(c).tolist())
        prepared.append((tag, name, move_lbl, sc, c))

    all_abs = np.sort([a for a in all_abs if a > 0])[::-1]
    shared_vmax = all_abs[0] if len(all_abs) else 1.0
    if shared_vmax <= 0:
        shared_vmax = 1.0

    # One row of three so the full distribution fits on a single manuscript page
    # (stacked 3x1 at full cell size overflows \textheight and clips the third panel).
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), dpi=180)

    for ax, item in zip(axes, prepared):
        if item is None:
            ax.text(0.5, 0.5, "missing data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            continue
        tag, name, move_lbl, sc, c = item
        M = vec_to_mat(c)
        plot_pair_heatmap(
            ax, M, shared_vmax,
            f"{name}\n[{tag}]  {move_lbl}\nself-sim = {sc:+.2f}",
            cmap=PLAYER_SHIFT_CMAP,
            cbar_label="shared displace\n(navy=weaken, coral=strengthen)",
            tick_fs=7, label_fs=8, title_fs=9, cbar_fs=6.5,
        )
        ax.title.set_color(CCOLORS[tag])

    fig.suptitle(
        "Portability is a distribution, not an invariant trait\n"
        "Cell colored only if origin & destination agree on sign; intensity = shared strength "
        "(unit-norm). Cream = no match.\n"
        "Gafford: sparse/deep cut links (upper-tail Δ).  "
        "Jones Jr.: many weaker mixed matches (median Δ).  "
        "Wright: little portable structure (lower-tail Δ).",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIG_DIR / "fig5b_signatures.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig5b_signatures.png")


# ----------------------------------------------------------------- Figure 7
# ----------------------------------------------------------------- Shaper Δ distribution
def figure_shaper_dist():
    """Histogram/density of shaper-mover Δ with Gafford / Jones Jr. / Wright marked."""
    u_med = TF["origin_usage"].median()
    # principled shaper cut: low origin usage × positive residual influence
    # (CSV role strings remain Shaper; do not rewrite output tables)
    conn = TF[(TF["origin_usage"] < u_med) & (TF["origin_influence_resid"] > 0)].copy()
    d = conn["delta"].to_numpy(float)
    med = float(np.median(d))

    marks = [
        ("Gafford", "D. Gafford", "WAS", "2023-24", "#1a6b3c"),
        ("Jones Jr.", "D. Jones Jr.", "CHI", "2022-23", "#2f5f8f"),
        ("Wright", "D. Wright", "WAS", "2023-24", "#666666"),
    ]
    mark_x = {}
    for label, player, team, season, color in marks:
        r = conn[(conn["player"] == player) & (conn["origin_team"] == team)
                 & (conn["origin_season"] == season)]
        if r.empty:
            print(f"  warn: missing mark {label}")
            continue
        mark_x[label] = (float(r.iloc[0]["delta"]), color)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    ax.hist(d, bins=22, density=True, color="#c5cdd8", edgecolor="white",
            linewidth=0.6, alpha=0.95, label=None)
    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(np.min(d) - 0.05, np.max(d) + 0.05, 300)
        ax.plot(xs, gaussian_kde(d)(xs), color="#2f5f8f", lw=1.8)
    except Exception:
        pass

    ax.axvline(0.0, color="#222", lw=1.1, ls="--", zorder=3)
    ax.axvline(med, color="#2f5f8f", lw=1.2, ls=":", zorder=3)

    ymax = ax.get_ylim()[1]
    heights = {"Gafford": 0.92, "Jones Jr.": 0.72, "Wright": 0.52}
    for label, (x, color) in mark_x.items():
        ax.axvline(x, color=color, lw=1.6, zorder=4)
        ax.annotate(
            f"{label}\nΔ={x:+.2f}",
            xy=(x, heights[label] * ymax),
            xytext=(8 if label != "Wright" else -8, 0),
            textcoords="offset points",
            ha="left" if label != "Wright" else "right",
            va="center", fontsize=8, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    ax.text(0.02, 0.97, f"n={len(d)} shaper movers\nmedian Δ={med:+.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))
    ax.set_xlabel(r"portability excess $\Delta$ (self-sim $-$ decoy-sim)")
    ax.set_ylabel("density")
    ax.set_title("Shaper portability is a distribution, not an invariant", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "fig_shaper_delta_dist.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


# ----------------------------------------------------------------- Taxonomy map
def figure_taxonomy_exemplars():
    """Usage percentile x residual percentile + hex density.

    Labels tell one story: concentrically organized hubs (Organizer / Shaper)
    vs high-usage volume that does not reorganize wiring beyond usage (Terminal).
    LAL 2023-24 is the within-roster contrast (AD hub, LeBron volume).
    """
    roles = pd.read_csv(OUT_DIR / "structural_role_classification.csv").copy()
    resid = roles["influence_residual"].to_numpy(float)
    share_pos = float((resid > 0).mean())

    roles["u_pct"] = roles["usage"].rank(pct=True) * 100
    roles["r_pct"] = roles["influence_residual"].rank(pct=True) * 100
    resid0_pct = float((resid <= 0).mean() * 100)

    # Same seasons as Sec. emerge prose / Table exemplars.
    # First tuple field is display class only; CSV role strings stay Shaper.
    exemplars = [
        ("Organizer", "SGA", "S. Gilgeous-Alexander", "OKC", "2023-24", "#1a6b3c"),
        ("Organizer", "Jokić", "N. Jokić", "DEN", "2024-25", "#1a6b3c"),
        ("Organizer", "AD", "A. Davis", "LAL", "2023-24", "#1a6b3c"),
        ("Shaper", "Gobert", "R. Gobert", "MIN", "2023-24", "#2f5f8f"),
        ("Shaper", "Gafford", "D. Gafford", "WAS", "2022-23", "#2f5f8f"),
        ("Terminal", "LeBron", "L. James", "LAL", "2023-24", "#c0392b"),
        ("Terminal", "Tatum", "J. Tatum", "BOS", "2022-23", "#c0392b"),
        ("Specialist", "Hart", "J. Hart", "NYK", "2023-24", "#6b6b6b"),
    ]

    dens_cmap = LinearSegmentedColormap.from_list(
        "taxonomy_density",
        ["#f7f4ef", "#d9d2c5", "#a8b0b8", "#5f6f7d", "#2c3a47"],
    )

    fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=180)
    hb = ax.hexbin(
        roles["u_pct"], roles["r_pct"],
        gridsize=26, mincnt=1, cmap=dens_cmap,
        linewidths=0.15, edgecolors="#efece5", zorder=1,
    )

    ax.axvline(50, color="#222", lw=1.15, ls="--", zorder=3)
    ax.axhline(resid0_pct, color="#222", lw=1.4, ls="--", zorder=3)
    ax.text(50, 1.5, "usage\nmedian", ha="center", va="bottom", fontsize=7.5, color="#222")
    ax.text(99, resid0_pct - 1.5, f"resid=0\n({resid0_pct:.0f}th pctile)",
            ha="right", va="top", fontsize=7.5, color="#222")

    box_kw = dict(fontsize=8, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.92))
    ax.text(55, 86, "Organizer\n(on-ball halfcourt)", transform=ax.transData,
            ha="left", va="top", color="#1a6b3c",
            **{**box_kw, "bbox": {**box_kw["bbox"], "ec": "#1a6b3c"}})
    ax.text(3, 93, "Shaper\n(cut-and-roll, low usage)", transform=ax.transData,
            ha="left", va="top", color="#2f5f8f",
            **{**box_kw, "bbox": {**box_kw["bbox"], "ec": "#2f5f8f"}})
    ax.text(55, 12, "Terminal Scorer\n(creation volume, replaceable wiring)", transform=ax.transData,
            ha="left", va="bottom", color="#c0392b",
            **{**box_kw, "bbox": {**box_kw["bbox"], "ec": "#c0392b"}})
    ax.text(3, 2, "Role Occ. / Specialist\n(mostly spot-ups)", transform=ax.transData,
            ha="left", va="bottom", color="#6b6b6b",
            **{**box_kw, "bbox": {**box_kw["bbox"], "ec": "#6b6b6b"}})

    coords = {}
    offsets = {
        "SGA": (-10, 14), "Jokić": (12, 10), "AD": (-8, 16),
        "Gobert": (-14, 14), "Gafford": (10, 10),
        "LeBron": (14, 22), "Tatum": (-48, 14),
        "Hart": (-36, -14),
    }

    for cls, short, player, team, season, color in exemplars:
        hit = roles[(roles["player"] == player) & (roles["team"] == team)
                    & (roles["season"] == season)]
        if hit.empty:
            print(f"  warn: missing exemplar {player} {team} {season}")
            continue
        x = float(hit.iloc[0]["u_pct"])
        y = float(hit.iloc[0]["r_pct"])
        coords[short] = (x, y)
        ax.scatter([x], [y], s=78, color=color, edgecolor="black",
                   linewidth=0.65, zorder=5, alpha=0.95)
        dx, dy = offsets.get(short, (8, 8))
        ax.annotate(
            short, (x, y), textcoords="offset points", xytext=(dx, dy),
            fontsize=8.5, color=color, fontweight="bold",
            ha="left" if dx >= 0 else "right",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92),
            zorder=6, annotation_clip=False,
        )

    if "AD" in coords and "LeBron" in coords:
        x1, y1 = coords["AD"]
        x2, y2 = coords["LeBron"]
        ax.plot([x1, x2], [y1, y2], color="#555", lw=0.9, ls=":", zorder=4)
        ax.annotate(
            "same Lakers roster, 2023-24:\nAD = post/cut/PnR organizer\nLeBron = drive/PnR volume",
            xy=((x1 + x2) / 2, (y1 + y2) / 2),
            xytext=(6, 42), textcoords="data",
            fontsize=7.5, color="#333", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#555", lw=0.7),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#555", alpha=0.95),
            zorder=7,
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("usage percentile")
    ax.set_ylabel("residual structural influence percentile")
    ax.set_title(
        "Organizer vs Terminal: does removal reorganize initiation pathways?\n"
        f"(hex = rotation-pool density, n={len(roles)}; only {share_pos*100:.0f}% clear resid>0)",
        fontsize=10.5,
    )
    cbar = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("player-seasons in bin\n(pool density, not an exemplar)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "fig_taxonomy_map.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure1()
    figure5b()
    figure_shaper_dist()
    figure_taxonomy_exemplars()
    figure_loo_displacement()
    print(f"\nPaper figures ({len(list(FIG_DIR.glob('fig*.png')))} files) -> {FIG_DIR}")
