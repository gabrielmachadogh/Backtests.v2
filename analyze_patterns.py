#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object


def ensure_results_dir():
    os.makedirs("results", exist_ok=True)


def format_pct_ptbr(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    s = f"{x:.{decimals}f}".replace(".", ",")
    return f"{s}%"


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
        "win_rate_pct", "win_rate_pct_num",
        "signal_time_dt",
    }

    numeric_cols = []
    for c in trades_df.columns:
        if c in ignore:
            continue
        if pd.api.types.is_numeric_dtype(trades_df[c]):
            numeric_cols.append(c)
    return numeric_cols


def time_split(df: pd.DataFrame, time_col: str, test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return df.copy(), df.copy()
    cut = int(np.floor(n * (1.0 - test_frac)))
    cut = max(1, min(cut, n - 1))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test


def compute_threshold(series_train: pd.Series, mode: str, pct: int) -> Optional[float]:
    x = series_train.dropna()
    if len(x) < 20:
        return None
    if x.nunique() < 2:
        return None

    if mode == "low":
        return float(x.quantile(pct / 100.0))
    if mode == "high":
        return float(x.quantile(1.0 - pct / 100.0))
    raise ValueError("mode must be 'low' or 'high'")


def apply_filter(df: pd.DataFrame, feature: str, mode: str, threshold: float) -> pd.DataFrame:
    if mode == "low":
        return df[df[feature] <= threshold]
    if mode == "high":
        return df[df[feature] >= threshold]
    raise ValueError("mode must be 'low' or 'high'")


def oos_univariate(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    baseline_train_wr: float,
    baseline_test_wr: float,
    features: List[str],
    percentiles: List[int],
    min_trades_train: int,
    min_trades_test: int,
    rr: float,
    timeframe: str,
    setup: str,
) -> pd.DataFrame:
    rows = []

    for feat in features:
        s_train = df_train[feat]
        if s_train.dropna().shape[0] < min_trades_train:
            continue

        for pct in percentiles:
            for mode in ["low", "high"]:
                thr = compute_threshold(s_train, mode=mode, pct=pct)
                if thr is None:
                    continue

                sub_train = apply_filter(df_train, feat, mode, thr)
                sub_test = apply_filter(df_test, feat, mode, thr)

                t_tr, w_tr, l_tr, wr_tr = stats_from_outcome(sub_train)
                t_te, w_te, l_te, wr_te = stats_from_outcome(sub_test)

                if t_tr < min_trades_train or t_te < min_trades_test:
                    continue

                rows.append({
                    "timeframe": timeframe,
                    "setup": setup,
                    "rr": rr,
                    "feature": feat,
                    "mode": mode,
                    "pct": pct,
                    "threshold": thr,
                    "trades_train": t_tr,
                    "win_rate_train_num": wr_tr,
                    "win_rate_train": format_pct_ptbr(wr_tr, 1),
                    "delta_vs_base_train_num": wr_tr - baseline_train_wr,
                    "delta_vs_base_train": format_pct_ptbr(wr_tr - baseline_train_wr, 1),

                    "trades_test": t_te,
                    "win_rate_test_num": wr_te,
                    "win_rate_test": format_pct_ptbr(wr_te, 1),
                    "delta_vs_base_test_num": wr_te - baseline_test_wr,
                    "delta_vs_base_test": format_pct_ptbr(wr_te - baseline_test_wr, 1),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        ["delta_vs_base_test_num", "trades_test", "win_rate_test_num"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return out


def pick_non_inconclusive_features(
    uni_df: pd.DataFrame,
    min_improvement_pp: float,
    max_features: int,
) -> List[str]:
    if uni_df.empty:
        return []

    g = uni_df.groupby("feature", dropna=False).agg({
        "delta_vs_base_test_num": "max",
        "trades_test": "max",
        "win_rate_test_num": "max",
    }).reset_index()

    g = g.sort_values(["delta_vs_base_test_num", "trades_test"], ascending=[False, False])
    g = g[g["delta_vs_base_test_num"] >= min_improvement_pp]
    return g["feature"].tolist()[:max_features]


def compute_inconclusive_features(
    uni_df: pd.DataFrame,
    all_features: List[str],
    min_improvement_pp: float,
) -> pd.DataFrame:
    if uni_df.empty:
        return pd.DataFrame({"feature": all_features, "best_delta_test_pp": [np.nan]*len(all_features), "status": ["inconclusive"]*len(all_features)})

    best = uni_df.groupby("feature")["delta_vs_base_test_num"].max().to_dict()
    rows = []
    for f in all_features:
        v = best.get(f, None)
        if v is None or (not np.isfinite(v)) or v < min_improvement_pp:
            rows.append({"feature": f, "best_delta_test_pp": v, "status": "inconclusive"})
        else:
            rows.append({"feature": f, "best_delta_test_pp": v, "status": "candidate"})
    return pd.DataFrame(rows).sort_values(["status", "best_delta_test_pp"], ascending=[True, False]).reset_index(drop=True)


def dedup_features_by_hash(train_df: pd.DataFrame, features: List[str]) -> Tuple[List[str], pd.DataFrame]:
    """
    Remove features redundantes (idênticas) no TREINO via hash.
    Retorna (features_unicas, df_relatorio_redundancia).
    """
    sig_map: Dict[int, str] = {}
    keep: List[str] = []
    rows = []

    for f in features:
        s = train_df[f]
        try:
            h = int(hash_pandas_object(s, index=False).sum())
        except Exception:
            keep.append(f)
            continue

        if h not in sig_map:
            sig_map[h] = f
            keep.append(f)
        else:
            master = sig_map[h]
            rows.append({
                "feature_kept": master,
                "feature_dropped": f,
                "reason": "identical_hash_on_train",
            })

    rep = pd.DataFrame(rows)
    return keep, rep


def oos_pairwise_fixed_grid(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    baseline_train_wr: float,
    baseline_test_wr: float,
    features: List[str],
    rr: float,
    timeframe: str,
    setup: str,
    min_trades_train: int,
    min_trades_test: int,
) -> pd.DataFrame:
    grid = [("low", 20), ("low", 30), ("high", 20), ("high", 30)]
    rows = []
    thr_cache: Dict[Tuple[str, str, int], Optional[float]] = {}

    def get_thr(feat: str, mode: str, pct: int) -> Optional[float]:
        k = (feat, mode, pct)
        if k in thr_cache:
            return thr_cache[k]
        thr_cache[k] = compute_threshold(df_train[feat], mode, pct)
        return thr_cache[k]

    for a, b in itertools.combinations(features, 2):
        for (mode_a, pct_a) in grid:
            thr_a = get_thr(a, mode_a, pct_a)
            if thr_a is None:
                continue
            train_a = apply_filter(df_train, a, mode_a, thr_a)
            test_a = apply_filter(df_test, a, mode_a, thr_a)

            for (mode_b, pct_b) in grid:
                thr_b = get_thr(b, mode_b, pct_b)
                if thr_b is None:
                    continue
                train_ab = apply_filter(train_a, b, mode_b, thr_b)
                test_ab = apply_filter(test_a, b, mode_b, thr_b)

                t_tr, w_tr, l_tr, wr_tr = stats_from_outcome(train_ab)
                t_te, w_te, l_te, wr_te = stats_from_outcome(test_ab)

                if t_tr < min_trades_train or t_te < min_trades_test:
                    continue

                rows.append({
                    "timeframe": timeframe,
                    "setup": setup,
                    "rr": rr,
                    "feature_a": a,
                    "mode_a": mode_a,
                    "pct_a": pct_a,
                    "threshold_a": thr_a,
                    "feature_b": b,
                    "mode_b": mode_b,
                    "pct_b": pct_b,
                    "threshold_b": thr_b,

                    "trades_train": t_tr,
                    "win_rate_train_num": wr_tr,
                    "win_rate_train": format_pct_ptbr(wr_tr, 1),
                    "delta_vs_base_train_num": wr_tr - baseline_train_wr,
                    "delta_vs_base_train": format_pct_ptbr(wr_tr - baseline_train_wr, 1),

                    "trades_test": t_te,
                    "win_rate_test_num": wr_te,
                    "win_rate_test": format_pct_ptbr(wr_te, 1),
                    "delta_vs_base_test_num": wr_te - baseline_test_wr,
                    "delta_vs_base_test": format_pct_ptbr(wr_te - baseline_test_wr, 1),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        ["delta_vs_base_test_num", "trades_test", "win_rate_test_num"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return out


def write_oos_best_md(
    baseline_df: pd.DataFrame,
    uni_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    out_path: str,
    top_n: int = 30,
):
    lines = []
    lines.append("# OOS BEST (PFR 1h)\n\n")
    lines.append("Split temporal (treino/teste). Thresholds calculados **só no treino**.\n\n")

    lines.append("## Baseline (PFR cru)\n\n")
    if baseline_df.empty:
        lines.append("(sem baseline)\n\n")
    else:
        lines.append("| rr | trades_train | wr_train | trades_test | wr_test |\n")
        lines.append("|---:|---:|---:|---:|---:|\n")
        for _, r in baseline_df.iterrows():
            lines.append(f"| {r['rr']} | {int(r['trades_train'])} | {r['win_rate_train']} | {int(r['trades_test'])} | {r['win_rate_test']} |\n")
        lines.append("\n")

    lines.append("## Top univariado (OOS)\n\n")
    if uni_df.empty:
        lines.append("(sem univariado)\n\n")
    else:
        top = uni_df.head(top_n)
        lines.append("| rr | feature | mode | pct | threshold | trades_test | wr_test | delta_test |\n")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for _, r in top.iterrows():
            lines.append(
                f"| {r['rr']} | {r['feature']} | {r['mode']} | {int(r['pct'])} | {r['threshold']:.6g} | {int(r['trades_test'])} | {r['win_rate_test']} | {r['delta_vs_base_test']} |\n"
            )
        lines.append("\n")

    lines.append("## Top pairwise (OOS, grade fixa)\n\n")
    if pair_df.empty:
        lines.append("(sem pairwise)\n\n")
    else:
        top = pair_df.head(top_n)
        lines.append("| rr | feature_a | a | feature_b | b | trades_test | wr_test | delta_test |\n")
        lines.append("|---:|---|---:|---|---:|---:|---:|---:|\n")
        for _, r in top.iterrows():
            a = f"{r['mode_a']}{int(r['pct_a'])}@{r['threshold_a']:.6g}"
            b = f"{r['mode_b']}{int(r['pct_b'])}@{r['threshold_b']:.6g}"
            lines.append(
                f"| {r['rr']} | {r['feature_a']} | {a} | {r['feature_b']} | {b} | {int(r['trades_test'])} | {r['win_rate_test']} | {r['delta_vs_base_test']} |\n"
            )
        lines.append("\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=str, required=True)
    parser.add_argument("--setup", type=str, default="PFR")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--min_trades_train", type=int, default=120)
    parser.add_argument("--min_trades_test", type=int, default=40)
    parser.add_argument("--min_improvement_pp", type=float, default=1.0)
    parser.add_argument("--max_pairwise_features", type=int, default=20)
    parser.add_argument("--percentiles", type=str, default="10,15,20,25,30,40,50,60")
    args = parser.parse_args()

    ensure_results_dir()

    trades_df = pd.read_csv(args.trades)

    if trades_df.empty:
        pd.DataFrame().to_csv("results/oos_baseline_PFR_1h.csv", index=False)
        pd.DataFrame().to_csv("results/oos_univariate_PFR_1h.csv", index=False)
        pd.DataFrame().to_csv("results/oos_pairwise_PFR_1h.csv", index=False)
        pd.DataFrame().to_csv("results/oos_inconclusive_features_PFR_1h.csv", index=False)
        pd.DataFrame().to_csv("results/oos_redundant_features_PFR_1h.csv", index=False)
        with open("results/oos_best_PFR_1h.md", "w", encoding="utf-8") as f:
            f.write("# OOS BEST (PFR 1h)\n\n(no trades)\n")
        print("No trades. Saved empty OOS reports.")
        return

    if "outcome" not in trades_df.columns:
        raise ValueError("CSV must contain 'outcome' column with values 'win'/'loss'.")
    if "signal_time" not in trades_df.columns:
        raise ValueError("Missing signal_time in trades CSV.")

    trades_df["signal_time_dt"] = pd.to_datetime(trades_df["signal_time"], utc=True, errors="coerce")
    trades_df["rr"] = pd.to_numeric(trades_df["rr"], errors="coerce")
    trades_df = trades_df.dropna(subset=["signal_time_dt", "rr"]).copy()

    if "setup" in trades_df.columns:
        trades_df = trades_df[trades_df["setup"] == args.setup].copy()
    if "timeframe" in trades_df.columns:
        trades_df = trades_df[trades_df["timeframe"] == args.timeframe].copy()

    if trades_df.empty:
        raise ValueError("After filters (setup/timeframe), there are 0 trades.")

    features_all = get_feature_columns(trades_df)
    percentiles = [int(x.strip()) for x in args.percentiles.split(",") if x.strip()]

    baseline_rows = []
    uni_rows_all = []
    pair_rows_all = []
    redund_rows_all = []

    for rr_val, rr_grp in trades_df.groupby("rr", dropna=False):
        rr_grp = rr_grp.sort_values("signal_time_dt").reset_index(drop=True)
        train, test = time_split(rr_grp, "signal_time_dt", test_frac=args.test_frac)

        t_tr, w_tr, l_tr, wr_tr = stats_from_outcome(train)
        t_te, w_te, l_te, wr_te = stats_from_outcome(test)

        baseline_rows.append({
            "timeframe": args.timeframe,
            "setup": args.setup,
            "rr": rr_val,
            "trades_train": t_tr,
            "win_rate_train_num": wr_tr,
            "win_rate_train": format_pct_ptbr(wr_tr, 1),
            "trades_test": t_te,
            "win_rate_test_num": wr_te,
            "win_rate_test": format_pct_ptbr(wr_te, 1),
        })

        uni = oos_univariate(
            df_train=train,
            df_test=test,
            baseline_train_wr=wr_tr,
            baseline_test_wr=wr_te,
            features=features_all,
            percentiles=percentiles,
            min_trades_train=args.min_trades_train,
            min_trades_test=args.min_trades_test,
            rr=rr_val,
            timeframe=args.timeframe,
            setup=args.setup,
        )
        if not uni.empty:
            uni_rows_all.append(uni)

        if not uni.empty:
            candidates = pick_non_inconclusive_features(
                uni_df=uni,
                min_improvement_pp=args.min_improvement_pp,
                max_features=args.max_pairwise_features,
            )
        else:
            candidates = []

        candidates_dedup, rep = dedup_features_by_hash(train, candidates)
        if not rep.empty:
            rep.insert(0, "rr", rr_val)
            redund_rows_all.append(rep)

        pair = oos_pairwise_fixed_grid(
            df_train=train,
            df_test=test,
            baseline_train_wr=wr_tr,
            baseline_test_wr=wr_te,
            features=candidates_dedup,
            rr=rr_val,
            timeframe=args.timeframe,
            setup=args.setup,
            min_trades_train=args.min_trades_train,
            min_trades_test=args.min_trades_test,
        )
        if not pair.empty:
            pair_rows_all.append(pair)

    baseline_df = pd.DataFrame(baseline_rows).sort_values(["rr"]).reset_index(drop=True)
    uni_df = pd.concat(uni_rows_all, ignore_index=True) if uni_rows_all else pd.DataFrame()
    pair_df = pd.concat(pair_rows_all, ignore_index=True) if pair_rows_all else pd.DataFrame()
    redund_df = pd.concat(redund_rows_all, ignore_index=True) if redund_rows_all else pd.DataFrame(columns=["rr", "feature_kept", "feature_dropped", "reason"])

    inconc_df = compute_inconclusive_features(
        uni_df=uni_df,
        all_features=features_all,
        min_improvement_pp=args.min_improvement_pp,
    )

    baseline_path = "results/oos_baseline_PFR_1h.csv"
    uni_path = "results/oos_univariate_PFR_1h.csv"
    pair_path = "results/oos_pairwise_PFR_1h.csv"
    inconc_path = "results/oos_inconclusive_features_PFR_1h.csv"
    redund_path = "results/oos_redundant_features_PFR_1h.csv"
    best_md_path = "results/oos_best_PFR_1h.md"

    baseline_df.to_csv(baseline_path, index=False)
    uni_df.to_csv(uni_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    inconc_df.to_csv(inconc_path, index=False)
    redund_df.to_csv(redund_path, index=False)

    write_oos_best_md(
        baseline_df=baseline_df,
        uni_df=uni_df,
        pair_df=pair_df,
        out_path=best_md_path,
        top_n=30,
    )

    print(f"Saved baseline:     {baseline_path} ({len(baseline_df)} rows)")
    print(f"Saved univariate:   {uni_path} ({len(uni_df)} rows)")
    print(f"Saved pairwise:     {pair_path} ({len(pair_df)} rows)")
    print(f"Saved inconclusive: {inconc_path} ({len(inconc_df)} rows)")
    print(f"Saved redundant:    {redund_path} ({len(redund_df)} rows)")
    print(f"Saved best md:      {best_md_path}")


if __name__ == "__main__":
    main()
