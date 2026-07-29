"""
Origin cut vs drive diet within connector movers (paper Sec. travelfit).

Reads frozen kit tables only (no Neo4j). Writes a one-row summary CSV with the
reported Spearman / AUC / tercile statistics.

Run: python cut_drive_travel_screen.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parent / "output"


def auc_of(pred, truth) -> float:
    pred = np.asarray(pred, float)
    truth = np.asarray(truth, bool)
    pos, neg = pred[truth], pred[~truth]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(
        np.mean(
            [
                1.0 if x > y else 0.5 if x == y else 0.0
                for x, y in product(pos, neg)
            ]
        )
    )


def main() -> None:
    tf = pd.read_csv(OUT / "travel_fit_screen.csv", dtype={"player_id": str})
    sig = pd.read_parquet(OUT / "portability_signatures.parquet")
    sig["player_id"] = sig["player_id"].astype(str)

    u_med = tf["origin_usage"].median()
    tf["is_conn"] = (tf["origin_usage"] < u_med) & (tf["origin_load_resid"] > 0)
    tf["portable"] = tf["delta"] > 0
    tf["origin_yy"] = tf["origin_season"].map(
        lambda s: int(str(s).split("-")[0][2:])
    )
    tf["origin_key"] = tf["origin_team"] + "_" + tf["origin_yy"].astype(str)

    so = sig.set_index(["player_id", "key"])
    cut, drv = [], []
    for r in tf.itertuples():
        try:
            row = so.loc[(r.player_id, r.origin_key)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            cut.append(float(row["p5"]))
            drv.append(float(row["p2"]))
        except Exception:
            cut.append(np.nan)
            drv.append(np.nan)
    tf["cut_share"] = cut
    tf["drive_share"] = drv
    tf["cut_minus_drive"] = tf["cut_share"] - tf["drive_share"]

    conn = tf[tf["is_conn"]].dropna(subset=["cut_share", "delta"]).copy()
    rho, p = stats.spearmanr(conn["cut_share"], conn["delta"])
    auc = auc_of(conn["cut_share"], conn["portable"])
    rho_cd, p_cd = stats.spearmanr(conn["cut_minus_drive"], conn["delta"])

    conn["cut_ter"] = pd.qcut(conn["cut_share"], 3, labels=["low", "mid", "high"])
    hi = conn[conn["cut_ter"] == "high"]
    lo = conn[conn["cut_ter"] == "low"]

    summary = pd.DataFrame(
        [
            {
                "n_connectors": len(conn),
                "cut_rho": float(rho),
                "cut_p": float(p),
                "cut_auc": float(auc),
                "cut_minus_drive_rho": float(rho_cd),
                "cut_minus_drive_p": float(p_cd),
                "high_cut_pct_portable": float(100 * hi["portable"].mean()),
                "high_cut_median_delta": float(hi["delta"].median()),
                "low_cut_pct_portable": float(100 * lo["portable"].mean()),
                "low_cut_median_delta": float(lo["delta"].median()),
            }
        ]
    )
    out = OUT / "cut_drive_travel_screen.csv"
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
