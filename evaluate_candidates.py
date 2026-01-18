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
    """
    Expectancy em unidades de R:
      EV = P(win)*RR - P(loss)*1
    """
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
    mode: str   # "low" | "high"
    pct: int    # quantil usado no TREINO


@dataclass
class CandidateSpec:
    name: str
    rr: float
    filters: List[FilterSpec]


def build_default_candidates() -> List[CandidateSpec]:
    """
    Conjunto pequeno 
