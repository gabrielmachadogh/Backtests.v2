#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def format_pct_ptbr(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    s = f"{x:.{decimals}f}".replace(".", ",")
    return f"{s}%"


def is_binary_like(s: pd.Series) -> bool:
    v = s.dropna().unique()
    if len(v) == 0:
        return False
    if len(v) <= 2:
        # allow {0,1}, {False,True}, etc.
        return True
    return False


def make_bins_quartiles(s: pd.Series) -> Optional[pd.Series]:
    """
    Returns bin labels 0..3 using qcut (quartiles). If not possible, returns None.
    """
    x = s.dropna()
    if len(x) < 10:
        return None
    if x.nunique() < 4:
        return None
    try:
        bins = pd.qcut(x, 4, labels=[0, 1, 2, 3], duplicates="drop")
        # align back to original index
        out = pd.Series(index=s.index, dtype="float")
        out.loc[x.index] = bins.astype(float)
        return out
    except Exception:
        return None


def make_bins_for_pairwise(s: pd.Series) -> Optional[pd.Series]:
    """
    For pairwise:
    - if binary-like => use values (0/1) as bins
    - else => quartiles (0..3)
    """
    if is_binary_like(s):
        # cast to int bins (0/1)
        x = s.copy()
        # normalize booleans
        if x.dropna().dtype == bool:
            x = x.astype(int)
        return x
    return make_bins_quartiles(s)


def stats_from_outcome(df: pd.DataFrame) -> Tuple[int, int, int, float]:
    trades = int(len(df))
    wins = int((df["outcome"] == "win").sum())
    losses = trades - wins
    win_rate = (wins / trades) * 100.0 if trades > 0 else np.nan
    return trades, wins, losses, win_rate


def get_feature_columns(trades_df: pd.DataFrame) -> List[str]:
    ignore = {
        "timeframe", "symbol", "setup", "side",
        "signal_time", "entry_time", "exit_time",
        "signal_idx", "entry_idx", "exit_idx",
        "entry", "stop", "risk", "rr", "tp",
        "outcome", "exit_reason", "fill_delay", "bars_in_trade",
        # formatted columns if present
        "win_rate_pct", "win_rate_pct_num",
    }
    numeric_cols = []
    for c in trades_df.columns:
        if c in ignore:
            continue
        if pd.api.types.is_numeric_dtype(trades_df[c]):
            numeric_cols.append(c)
    return numeric_cols


def build_univariate_patterns(
    trades_df: pd.DataFrame,
    features: List[str],
    min_trades: int,
    pct_buckets: List[int] = [10, 15, 20, 25, 30, 40, 50, 60],
) -> pd.DataFrame:
    rows = []

    group_cols = ["timeframe", "setup", "rr"]
    for keys, grp in trades_df.groupby(group_cols, dropna=False):
        timeframe, setup, rr = keys
        grp = grp.copy()

        # ALL (baseline)
        trades, wins, losses, wr = stats_from_outcome(grp)
        rows.append({
            "timeframe": timeframe,
            "setup": setup,
            "rr": rr,
            "feature": "__ALL__",
            "bucket": "ALL",
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct_num": wr,
            "win_rate_pct": format_pct_ptbr(wr, 1),
        })

        for feat in features:
            s = grp[feat]
            if s.dropna().shape[0] < min_trades:
                continue

            # Quartiles
            qbins = make_bins_quartiles(s)
            if qbins is not None:
                tmp = grp.copy()
                tmp["bin"] = qbins
                tmp = tmp.dropna(subset=["bin"])
                for b, sub in tmp.groupby("bin", dropna=False):
                    t, w, l, wrb = stats_from_outcome(sub)
                    if t < min_trades:
                        continue
                    rows.append({
                        "timeframe": timeframe,
                        "setup": setup,
                        "rr": rr,
                        "feature": feat,
                        "bucket": f"quartile_{int(b)}",
                        "trades": t,
                        "wins": w,
                        "losses": l,
                        "win_rate_pct_num": wrb,
                        "win_rate_pct": format_pct_ptbr(wrb, 1),
                    })

            # Top/Low percent buckets
            x = s.dropna()
            if len(x) < min_trades:
                continue

            for p in pct_buckets:
                q_low = x.quantile(p / 100.0)
                q_high = x.quantile(1.0 - p / 100.0)

                low_sub = grp[grp[feat] <= q_low]
                top_sub = grp[grp[feat] >= q_high]

                for name, sub in [(f"low{p}", low_sub), (f"top{p}", top_sub)]:
                    t, w, l, wrb = stats_from_outcome(sub)
                    if t < min_trades:
                        continue
                    rows.append({
                        "timeframe": timeframe,
                        "setup": setup,
                        "rr": rr,
                        "feature": feat,
                        "bucket": name,
                        "trades": t,
                        "wins": w,
                        "losses": l,
                        "win_rate_pct_num": wrb,
                        "win_rate_pct": format_pct_ptbr(wrb, 1),
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["win_rate_pct_num", "trades"], ascending=[False, False]).reset_index(drop=True)
    return out


def build_pairwise_patterns(
    trades_df: pd.DataFrame,
    features: List[str],
    min_trades: int,
) -> pd.DataFrame:
    rows = []
    group_cols = ["timeframe", "setup", "rr"]

    for keys, grp in trades_df.groupby(group_cols, dropna=False):
        timeframe, setup, rr = keys
        grp = grp.copy()

        # precompute bins per feature for this group
        bins_map: Dict[str, Optional[pd.Series]] = {}
        for f in features:
            if grp[f].dropna().shape[0] < min_trades:
                bins_map[f] = None
                continue
            bins_map[f] = make_bins_for_pairwise(grp[f])

        feats_valid = [f for f in features if bins_map.get(f) is not None]

        for a, b in itertools.combinations(feats_valid, 2):
            ba = bins_map[a]
            bb = bins_map[b]
            if ba is None or bb is None:
                continue

            tmp = grp[["outcome"]].copy()
            tmp["feature_a"] = a
            tmp["feature_b"] = b
            tmp["bin_a"] = ba
            tmp["bin_b"] = bb
            tmp = tmp.dropna(subset=["bin_a", "bin_b"])

            if len(tmp) < min_trades:
                continue

            # group by bin combo
            gb = tmp.groupby(["bin_a", "bin_b"], dropna=False)
            for (bin_a, bin_b), sub in gb:
                t = int(len(sub))
                if t < min_trades:
                    continue
                wins = int((sub["outcome"] == "win").sum())
                losses = t - wins
                wr = (wins / t) * 100.0 if t > 0 else np.nan

                rows.append({
                    "timeframe": timeframe,
                    "setup": setup,
                    "rr": rr,
                    "feature_a": a,
                    "feature_b": b,
                    "bin_a": bin_a,
                    "bin_b": bin_b,
                    "trades": t,
                    "wins": wins,
                    "losses": losses,
                    "win_rate_pct_num": wr,
                    "win_rate_pct": format_pct_ptbr(wr, 1),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["win_rate_pct_num", "trades"], ascending=[False, False]).reset_index(drop=True)
    return out


def write_best_md(pairwise_df: pd.DataFrame, out_path: str, top_n: int = 30):
    if pairwise_df.empty:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# patterns_best\n\n(no patterns)\n")
        return

    lines = []
    lines.append("# patterns_best\n")

    group_cols = ["timeframe", "setup", "rr"]
    for keys, grp in pairwise_df.groupby(group_cols, dropna=False):
        timeframe, setup, rr = keys
        lines.append(f"## {timeframe} | {setup} | RR {rr}\n")
        gtop = grp.sort_values(["win_rate_pct_num", "trades"], ascending=[False, False]).head(top_n)

        lines.append("| feature_a | feature_b | bin_a | bin_b | trades | win_rate |\n")
        lines.append("|---|---:|---:|---:|---:|---:|\n")
        for _, r in gtop.iterrows():
            lines.append(
                f"| {r['feature_a']} | {r['feature_b']} | {r['bin_a']} | {r['bin_b']} | {int(r['trades'])} | {r['win_rate_pct']} |\n"
            )
        lines.append("\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=str, required=True)
    parser.add_argument("--min_trades", type=int, default=30)
    args = parser.parse_args()

    trades_df = pd.read_csv(args.trades)

    # ensure outcome exists
    if "outcome" not in trades_df.columns:
        raise ValueError("CSV must contain 'outcome' column with 'win'/'loss'.")

    # coerce rr to numeric
    trades_df["rr"] = pd.to_numeric(trades_df["rr"], errors="coerce")

    features = get_feature_columns(trades_df)

    uni = build_univariate_patterns(trades_df, features, min_trades=args.min_trades)
    pair = build_pairwise_patterns(trades_df, features, min_trades=args.min_trades)

    # Derive base name from input
    # Expected: results/backtest_trades_BTC_USDT_1h_long.csv
    base = args.trades.replace("backtest_trades_", "").replace(".csv", "")
    out_uni = f"results/patterns_univariate_{base}.csv"
    out_pair = f"results/patterns_pairwise_{base}.csv"
    out_best = f"results/patterns_best_{base}.md"

    uni.to_csv(out_uni, index=False)
    pair.to_csv(out_pair, index=False)
    write_best_md(pair, out_best, top_n=30)

    print(f"Saved univariate: {out_uni} ({len(uni)} rows)")
    print(f"Saved pairwise:   {out_pair} ({len(pair)} rows)")
    print(f"Saved best md:    {out_best}")


if __name__ == "__main__":
    main()
