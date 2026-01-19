#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def format_pct_ptbr(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    s = f"{x:.{decimals}f}".replace(".", ",")
    return f"{s}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="results/recommended")
    parser.add_argument("--splits", type=str, default="tf20,tf30,tf40")
    args = parser.parse_args()

    base_dir = args.base_dir
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    rows = []
    for s in splits:
        path = os.path.join(base_dir, s, "oos_candidates_report_PFR_1h.csv")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["split"] = s
        rows.append(df)

    if not rows:
        ensure_dir(base_dir)
        pd.DataFrame().to_csv(os.path.join(base_dir, "STABILITY.csv"), index=False)
        with open(os.path.join(base_dir, "STABILITY.md"), "w", encoding="utf-8") as f:
            f.write("# STABILITY\n\n(no data)\n")
        print("No candidate reports found.")
        return

    all_df = pd.concat(rows, ignore_index=True)

    # group by rr+name+rule (rule muda pouco, mas mantemos)
    gcols = ["rr", "name", "rule", "n_filters"]
    agg = all_df.groupby(gcols, dropna=False).agg(
        splits=("split", "nunique"),
        splits_pos_ev=("evR_test", lambda x: int(np.sum(np.array(x) > 0))),
        avg_trades_test=("trades_test", "mean"),
        min_trades_test=("trades_test", "min"),
        avg_wr_test_num=("wr_test_num", "mean"),
        avg_delta_wr_test_pp=("delta_wr_test_pp", "mean"),
        avg_evR_test=("evR_test", "mean"),
        avg_delta_evR_test=("delta_evR_test", "mean"),
    ).reset_index()

    # formatting
    agg["avg_wr_test"] = agg["avg_wr_test_num"].apply(lambda x: format_pct_ptbr(x, 1))
    agg["avg_delta_wr_test"] = agg["avg_delta_wr_test_pp"].apply(lambda x: format_pct_ptbr(x, 1))

    agg = agg.sort_values(
        ["splits_pos_ev", "avg_delta_evR_test", "avg_delta_wr_test_pp", "avg_trades_test"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    ensure_dir(base_dir)
    out_csv = os.path.join(base_dir, "STABILITY.csv")
    agg.to_csv(out_csv, index=False)

    # markdown
    out_md = os.path.join(base_dir, "STABILITY.md")
    lines = []
    lines.append("# STABILITY (PFR 1h)\n\n")
    lines.append("Ranking por estabilidade nos splits (tf20/tf30/tf40).\n\n")
    lines.append("| rr | name | filters | splits_pos_ev | splits | avg_trades_test | min_trades_test | avg_wr_test | avg_delta_wr | avg_evR_test | avg_delta_evR |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    top = agg.head(60)
    for _, r in top.iterrows():
        lines.append(
            f"| {r['rr']} | {r['name']} | {int(r['n_filters'])} | {int(r['splits_pos_ev'])} | {int(r['splits'])} | "
            f"{r['avg_trades_test']:.1f} | {int(r['min_trades_test'])} | {r['avg_wr_test']} | {r['avg_delta_wr_test']} | "
            f"{r['avg_evR_test']:.4f} | {r['avg_delta_evR_test']:.4f} |\n"
        )

    lines.append("\n## Observação\n\n")
    lines.append("- `splits_pos_ev` = em quantos splits o `evR_test` ficou > 0.\n")
    lines.append("- Prefira regras com `splits_pos_ev` alto e `min_trades_test` decente.\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
