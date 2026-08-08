"""Disaggregate the MARGINAL shaper tier by position: low-influence BIG vs
perimeter/SHOOTER (position has no 'C'). Three-way joint model:
  n_HUBCONN, n_MARGBIG, n_MARGWING  (+ n_ORGANIZER, n_TERMINAL controls)
Same team-season FE + opp/period/scorebucket spec throughout.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from wiring_gate import OUT_DIR
from role_composition_playmix import pull_possessions, LOADS, STATS, MIN_MPG, MIN_GP
from role_composition_ppp import dummies, demean_by_group, cluster_se, score_bucket


def two_sided_p(t):
    from math import erf, sqrt
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))


POS = pd.read_csv(OUT_DIR / "player_positions.csv")
pos_map = {(r.tri, int(r.yy), r.abbrev): r.position for r in POS.itertuples(index=False)}


def build_3way_class_map(season_yy=(22, 23, 24)):
    """Same taxonomy as role_composition_playmix.build_class_map() (Eq. 5 quadratic
    on graph-native u), then split Shaper/Hub at median residual influence and split
    the marginal half by position (big vs wing)."""
    loads = pd.read_csv(LOADS)
    stats = pd.read_csv(STATS)
    stats = stats.rename(columns={"abbrev": "player"})
    df = loads.merge(stats[["tri", "yy", "player", "usg_pct", "mpg", "gp"]],
                     on=["tri", "yy", "player"], how="left")
    df = df.copy()
    fit_df = df[df["yy"].isin(season_yy)] if season_yy is not None else df

    rot = fit_df[(fit_df["mpg"] >= MIN_MPG) & (fit_df["gp"] >= MIN_GP)
                & fit_df["usg_pct"].notna()].copy()
    u_med = float(rot["usg_pct"].median())

    uu = fit_df["u"].to_numpy(float)
    ll = fit_df["Lcos"].to_numpy(float)
    ok_fit = np.isfinite(uu) & np.isfinite(ll)
    X = np.column_stack([np.ones(int(ok_fit.sum())), uu[ok_fit], uu[ok_fit] ** 2])
    beta, *_ = np.linalg.lstsq(X, ll[ok_fit], rcond=None)

    ok = df["usg_pct"].notna() & df["Lcos"].notna() & df["u"].notna()
    resid = np.full(len(df), np.nan)
    xu = df.loc[ok, "u"].to_numpy(float)
    Xall = np.column_stack([np.ones(int(ok.sum())), xu, xu ** 2])
    resid[ok.to_numpy()] = df.loc[ok, "Lcos"].to_numpy(float) - Xall @ beta
    df["u_resid"] = resid
    if season_yy is not None:
        df = df[df["yy"].isin(season_yy)].copy()

    conn_mask = df["usg_pct"].notna() & df["Lcos"].notna() & df["u_resid"].notna() & \
        (df["usg_pct"] < u_med) & (df["u_resid"] > 0)
    r_med = float(df.loc[conn_mask, "u_resid"].median())

    cmap = {}
    unmatched = 0
    tot_marg = 0
    for r in df.itertuples(index=False):
        if not np.isfinite(r.Lcos) or not np.isfinite(r.usg_pct) or not np.isfinite(r.u_resid):
            continue
        hi_u = r.usg_pct >= u_med
        hi_r = r.u_resid > 0
        if hi_u and hi_r:
            cls = "ORGANIZER"
        elif hi_u and not hi_r:
            cls = "TERMINAL"
        elif (not hi_u) and not hi_r:
            cls = "OCCUPANT"
        else:
            if r.u_resid >= r_med:
                cls = "HUBCONN"
            else:
                tot_marg += 1
                pos = pos_map.get((r.tri, int(r.yy), r.player))
                if pos is None:
                    unmatched += 1
                    cls = "MARGBIG"  # default fallback; rare
                elif "C" in pos:
                    cls = "MARGBIG"
                else:
                    cls = "MARGWING"
        cmap[(r.tri, int(r.yy), r.player)] = cls
    print(f"marginal-tier players: {tot_marg}, unmatched position: {unmatched}")
    return cmap, u_med, r_med


ORDER5 = ["HUBCONN", "MARGBIG", "MARGWING", "ORGANIZER", "TERMINAL", "OCCUPANT"]


def add_composition6(df, cmap):
    plcols = [f"pl{i}" for i in range(5)]
    yy = df["yy"].to_numpy(); team = df["team"].to_numpy()
    comp = {k: np.zeros(len(df), dtype=np.int16) for k in ORDER5}
    ncl = np.zeros(len(df), dtype=np.int16)
    for c in plcols:
        names = df[c].to_numpy()
        arr = np.array([cmap.get((team[i], int(yy[i]), names[i]), "?") for i in range(len(df))],
                       dtype=object)
        for k in ORDER5:
            comp[k] += (arr == k)
        ncl += (arr != "?")
    for k in ORDER5:
        df[f"n_{k}"] = comp[k]
    df["n_classified"] = ncl
    return df


print("loading possessions + 3-way tiered role map...")
df = pull_possessions()
cmap, u_med, r_med = build_3way_class_map()

# show who's in each marginal subgroup, for a sanity check
rev = {}
for k, v in cmap.items():
    rev.setdefault(v, []).append(k)
print(f"HUBCONN n={len(rev.get('HUBCONN', []))}  MARGBIG n={len(rev.get('MARGBIG', []))}  "
      f"MARGWING n={len(rev.get('MARGWING', []))}")
print("MARGWING sample:", [k[2] for k in rev.get('MARGWING', [])][:15])
print("MARGBIG sample:", [k[2] for k in rev.get('MARGBIG', [])][:15])

df = add_composition6(df, cmap)
df = df[df["n_classified"] == 5].copy()
df["ts"] = df["team"] + "_" + df["yy"].astype(str)
df["sb"] = score_bucket(df["score_margin"].fillna(0).to_numpy()).astype(str)
df["per"] = df["period"].fillna(0).astype(int).astype(str)
df["has_hub"] = (df["n_HUBCONN"] >= 1).astype(int)
df["has_margbig"] = (df["n_MARGBIG"] >= 1).astype(int)
df["has_margwing"] = (df["n_MARGWING"] >= 1).astype(int)

print(f"\non-court subset (all 5 classified): n={len(df):,}, team-seasons={df['ts'].nunique()}")
for c in ["has_hub", "has_margbig", "has_margwing"]:
    print(f"  possessions with {c}=1: {df[c].mean()*100:.1f}%  (n={df[c].sum():,})")

ctrl = ["opp", "per", "sb"]
D = dummies(df, ctrl)
keycols = ["has_hub", "has_margbig", "has_margwing", "n_ORGANIZER", "n_TERMINAL"]
Xkey = df[keycols].astype(float)
Z = pd.concat([Xkey, D], axis=1)
znames = list(Z.columns)
y = df["pts"].to_numpy(float)
ts = df["ts"].to_numpy()

yb = (pd.Series(y) - pd.Series(y).groupby(pd.Series(ts)).transform("mean")).to_numpy()
Zb = demean_by_group(Z.to_numpy(float), ts)

beta, *_ = np.linalg.lstsq(Zb, yb, rcond=None)
resid = yb - Zb @ beta
se = cluster_se(Zb, resid, beta, ts)

print(f"\n{'term':<15}{'beta':>10}{'pts/100':>9}{'clustSE':>10}{'t':>7}{'p':>10}")
rows = []
for name in keycols:
    i = znames.index(name)
    b, s = beta[i], se[i]
    t = b / s if s > 0 else np.nan
    p = two_sided_p(t)
    print(f"{name:<15}{b:>10.4f}{b*100:>9.2f}{s:>10.4f}{t:>7.2f}{p:>10.2e}")
    rows.append(dict(term=name, beta=b, pts100=b*100, se=s, t=t, p=p,
                      n_on=int(df[name.replace('has_', 'has_') ].sum()) if name.startswith('has_') else None))
pd.DataFrame(rows).to_csv(OUT_DIR / "shaper_tier_ppp_3way.csv", index=False)
print(f"\nwrote {OUT_DIR / 'shaper_tier_ppp_3way.csv'}")
