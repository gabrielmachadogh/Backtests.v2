#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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


def expectancy_r(win_rate_pct: float, rr: float) -> float:
    if not np.isfinite(win_rate_pct):
        return np.nan
    p = win_rate_pct / 100.0
    return p * rr - (1 - p) * 1.0


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


@dataclass
class FilterSpec:
    feature: str
    mode: str
    pct: int


@dataclass
class CandidateSpec:
    name: str
    rr: float
    filters: List[FilterSpec]


def build_default_candidates() -> List[CandidateSpec]:
    cands: List[CandidateSpec] = []

    cands += [
        CandidateSpec("RR1_ret3_high50", 1.0, [FilterSpec("ret_3_pct", "high", 50)]),
        CandidateSpec("RR1_volz_high50", 1.0, [FilterSpec("vol_z", "high", 50)]),
        CandidateSpec("RR1_magap_high60", 1.0, [FilterSpec("ma_gap_pct", "high", 60)]),
        CandidateSpec("RR1_ret3_high50__AND__volz_high50", 1.0, [
            FilterSpec("ret_3_pct", "high", 50),
            FilterSpec("vol_z", "high", 50),
        ]),
        CandidateSpec("RR1_ret3_high50__AND__magap_high60", 1.0, [
            FilterSpec("ret_3_pct", "high", 50),
            FilterSpec("ma_gap_pct", "high", 60),
        ]),
    ]

    cands += [
        CandidateSpec("RR1p5_ret3_high50", 1.5, [FilterSpec("ret_3_pct", "high", 50)]),
        CandidateSpec("RR1p5_volz_high50", 1.5, [FilterSpec("vol_z", "high", 50)]),
        CandidateSpec("RR1p5_magap_high60", 1.5, [FilterSpec("ma_gap_pct", "high", 60)]),
        CandidateSpec("RR1p5_ret3_high50__AND__volz_high50", 1.5, [
            FilterSpec("ret_3_pct", "high", 50),
            FilterSpec("vol_z", "high", 50),
        ]),
    ]

    cands += [
        CandidateSpec("RR2_rsi_high60", 2.0, [FilterSpec("rsi", "high", 60)]),
        CandidateSpec("RR2_pullback_atr_low50", 2.0, [FilterSpec("pullback_from_new_high_atr", "low", 50)]),
        CandidateSpec("RR2_ret3_high50", 2.0, [FilterSpec("ret_3_pct", "high", 50)]),
        CandidateSpec("RR2_rsi_high60__AND__pullback_atr_low50", 2.0, [
            FilterSpec("rsi", "high", 60),
            FilterSpec("pullback_from_new_high_atr", "low", 50),
        ]),
        CandidateSpec("RR2_rsi_high60__AND__ret3_high50", 2.0, [
            FilterSpec("rsi", "high", 60),
            FilterSpec("ret_3_pct", "high", 50),
        ]),
        CandidateSpec("RR2_after_new_high_flag_high50", 2.0, [FilterSpec("after_new_high_flag", "high", 50)]),
        CandidateSpec("RR2_after_new_high_recent_flag_high50", 2.0, [FilterSpec("after_new_high_recent_flag", "high", 50)]),
    ]

    return cands


def evaluate_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rr: float,
    spec: CandidateSpec,
    min_trades_train: int,
    min_trades_test: int,
) -> Optional[Dict]:
    t0_tr, w0_tr, l0_tr, wr0_tr = stats_from_outcome(train)
    t0_te, w0_te, l0_te, wr0_te = stats_from_outcome(test)

    thresholds = []
    sub_tr = train
    sub_te = test

    for fs in spec.filters:
        if fs.feature not in train.columns:
            return None

        thr = compute_threshold(train[fs.feature], fs.mode, fs.pct)
        if thr is None:
            return None

        thresholds.append((fs.feature, fs.mode, fs.pct, thr))
        sub_tr = apply_filter(sub_tr, fs.feature, fs.mode, thr)
        sub_te = apply_filter(sub_te, fs.feature, fs.mode, thr)

    t_tr, w_tr, l_tr, wr_tr = stats_from_outcome(sub_tr)
    t_te, w_te, l_te, wr_te = stats_from_outcome(sub_te)

    if t_tr < min_trades_train or t_te < min_trades_test:
        return None

    ev0_tr = expectancy_r(wr0_tr, rr)
    ev0_te = expectancy_r(wr0_te, rr)
    ev_tr = expectancy_r(wr_tr, rr)
    ev_te = expectancy_r(wr_te, rr)

    rule_str = " AND ".join([f"{f} {m}{p}@{thr:.6g}" for (f, m, p, thr) in thresholds])

    return {
        "rr": rr,
        "name": spec.name,
        "rule": rule_str,
        "n_filters": len(spec.filters),

        "baseline_trades_train": t0_tr,
        "baseline_wr_train_num": wr0_tr,
        "baseline_wr_train": format_pct_ptbr(wr0_tr, 1),
        "baseline_evR_train": ev0_tr,

        "baseline_trades_test": t0_te,
        "baseline_wr_test_num": wr0_te,
        "baseline_wr_test": format_pct_ptbr(wr0_te, 1),
        "baseline_evR_test": ev0_te,

        "trades_train": t_tr,
        "wr_train_num": wr_tr,
        "wr_train": format_pct_ptbr(wr_tr, 1),
        "delta_wr_train_pp": wr_tr - wr0_tr,
        "delta_wr_train": format_pct_ptbr(wr_tr - wr0_tr, 1),
        "evR_train": ev_tr,
        "delta_evR_train": ev_tr - ev0_tr,

        "trades_test": t_te,
        "wr_test_num": wr_te,
        "wr_test": format_pct_ptbr(wr_te, 1),
        "delta_wr_test_pp": wr_te - wr0_te,
        "delta_wr_test": format_pct_ptbr(wr_te - wr0_te, 1),
        "evR_test": ev_te,
        "delta_evR_test": ev_te - ev0_te,
    }


def write_best_md(df: pd.DataFrame, out_path: str, top_n: int = 40):
    lines = []
    lines.append("# OOS CANDIDATES BEST (PFR 1h)\n\n")
    lines.append("Regras fixas (pré-definidas). Thresholds calculados no TREINO.\n\n")

    if df.empty:
        lines.append("(no candidates passed min trades)\n")
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return

    for rr, grp in df.groupby("rr", dropna=False):
        lines.append(f"## RR {rr}\n\n")
        g = grp.sort_values(["delta_evR_test", "delta_wr_test_pp", "trades_test"], ascending=[False, False, False]).head(top_n)

        lines.append("| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |\n")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|\n")
        for _, r in g.iterrows():
            lines.append(
                f"| {r['name']} | {int(r['n_filters'])} | {r['rule']} | {int(r['trades_test'])} | {r['wr_test']} | {r['delta_wr_test']} | {r['evR_test']:.4f} | {r['delta_evR_test']:.4f} |\n"
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
    args = parser.parse_args()

    ensure_results_dir()

    trades_df = pd.read_csv(args.trades)

    if trades_df.empty:
        pd.DataFrame().to_csv("results/oos_candidates_report_PFR_1h.csv", index=False)
        write_best_md(pd.DataFrame(), "results/oos_candidates_best_PFR_1h.md", top_n=40)
        print("No trades. Saved empty candidates reports.")
        return

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

    candidates = build_default_candidates()

    rows = []
    for rr_val, rr_grp in trades_df.groupby("rr", dropna=False):
        rr_grp = rr_grp.sort_values("signal_time_dt").reset_index(drop=True)
        train, test = time_split(rr_grp, "signal_time_dt", test_frac=args.test_frac)

        specs_rr = [c for c in candidates if float(c.rr) == float(rr_val)]
        for spec in specs_rr:
            res = evaluate_candidate(
                train=train,
                test=test,
                rr=float(rr_val),
                spec=spec,
                min_trades_train=args.min_trades_train,
                min_trades_test=args.min_trades_test,
            )
            if res is not None:
                rows.append(res)

    out = pd.DataFrame(rows)
    out_path = "results/oos_candidates_report_PFR_1h.csv"
    out.to_csv(out_path, index=False)

    md_path = "results/oos_candidates_best_PFR_1h.md"
    write_best_md(out, md_path, top_n=40)

    print(f"Saved candidates report: {out_path} ({len(out)} rows)")
    print(f"Saved candidates best:   {md_path}")


if __name__ == "__main__":
    main()
