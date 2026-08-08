"""
Shared machinery for the STRUCTURAL ROLE COVERAGE vs TALENT ACCUMULATION study.
Registered spec: TOPOLOGY_COVERAGE_PREREGISTRATION.md.

Provides:
  - add_covered(df)   : COVERED indicator (>=1 ORG & >=1 CONN & >=1 TERM) + role counts
  - build_talent(df)  : frozen individual-talent control TALENT_i =
                        mean over the 5 on-court of q_p, where q_p = each player's
                        season points-per-initiation, LEAVE-CURRENT-GAME-OUT.

Nothing here computes the coverage effect beta; that lives only in the confirmatory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from role_composition_playmix import ORDER
from role_composition_ppp import add_composition

MIN_INIT = 20                 # frozen: player needs >=20 season initiations, else impute
SEED = 17
N_BOOT = 2000
CONF_P = 0.01


def add_covered(df, cmap):
    """Add n_<CLASS> counts, n_classified, and COVERED (creation+connection+conversion)."""
    df = add_composition(df, cmap)
    df["COVERED"] = ((df["n_ORGANIZER"] >= 1) & (df["n_SHAPER"] >= 1)
                     & (df["n_TERMINAL"] >= 1)).astype(int)
    return df


def build_talent(df):
    """Frozen talent control. Returns TALENT_i array aligned to df.index order.

    q_p = (season initiation points - this-game initiation points)
          / (season initiations       - this-game initiations)   [leave-game-out]
    Players with <MIN_INIT season initiations get the team-season mean q imputed.
    TALENT_i = mean of q_p over the five on-court players.
    """
    team = df["team"].to_numpy()
    yy = df["yy"].to_numpy().astype(int)
    game = df["game_id"].to_numpy()

    # season + per-game initiation aggregates (only from possessions that HAVE an initiator)
    v = df[df["initiator"].notna()].copy()
    v["_k"] = list(zip(v["team"], v["yy"].astype(int), v["initiator"]))
    tot = v.groupby("_k")["pts"].agg(["sum", "count"])
    Spts = tot["sum"].to_dict()
    Scnt = tot["count"].to_dict()

    v["_kg"] = list(zip(v["team"], v["yy"].astype(int), v["initiator"], v["game_id"]))
    gg = v.groupby("_kg")["pts"].agg(["sum", "count"])
    Gpts = gg["sum"].to_dict()
    Gcnt = gg["count"].to_dict()

    # team-season mean q over players with >= MIN_INIT initiations (imputation target)
    qser = {k: Spts[k] / Scnt[k] for k in Spts if Scnt[k] >= MIN_INIT}
    ts_of = {k: (k[0], k[1]) for k in qser}
    tsdf = pd.DataFrame({"ts": [ts_of[k] for k in qser], "q": [qser[k] for k in qser]})
    tsmean = tsdf.groupby("ts")["q"].mean().to_dict()
    global_q = float(np.mean(list(qser.values()))) if qser else 0.0

    n = len(df)
    acc = np.zeros(n, dtype=float)
    plcols = [f"pl{i}" for i in range(5)]
    for c in plcols:
        names = df[c].to_numpy()
        keys = list(zip(team, yy, names))
        gkeys = list(zip(team, yy, names, game))
        q = np.empty(n, dtype=float)
        for i in range(n):
            k = keys[i]
            sc = Scnt.get(k, 0)
            if sc >= MIN_INIT:
                gp = Gpts.get(gkeys[i], 0.0)
                gc = Gcnt.get(gkeys[i], 0)
                denom = sc - gc
                q[i] = (Spts[k] - gp) / denom if denom > 0 else tsmean.get(ts_of.get(k, (team[i], yy[i])), global_q)
            else:
                q[i] = tsmean.get((team[i], yy[i]), global_q)
        acc += q
    return acc / 5.0
