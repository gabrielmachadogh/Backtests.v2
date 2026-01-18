#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest BTC_USDT perp (MEXC) - 1h - ONLY LONG - Setups: PFR + Dave Landry
Rules/Features as specified:
- Trend filter: close > SMA8 and close > SMA80 and SMA8 > SMA80 (on signal candle i)
- Slope filter (mandatory): SMA8[i] > max(SMA8[i-8:i])  (max of previous 8 SMA8 values)
- Entry: buy stop at high[i] + 1 tick
- Stop: low[i] - 1 tick
- MAX_ENTRY_WAIT_BARS = 1 (must trigger on i+1 or cancel)
- RR: 1.0, 1.5, 2.0, 3.0 (independent simulations)
- One position at a time
- No fees/slippage/funding
- Conservative ambiguity: if TP and SL hit same candle => LOSS

Outputs:
- results/backtest_trades_{symbol}_{timeframe}_long.csv
- results/backtest_summary_{symbol}_{timeframe}_long.csv
"""

import os
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests


# =========================
# Config
# =========================
MEXC_CONTRACT_BASE = "https://contract.mexc.com"

DEFAULT_TICK_SIZE = 0.1

ATR_PERIOD = 14
RSI_PERIOD = 14
STRUCT_N = 20
EXTREME_LOOKBACK = 20

AMBIGUOUS_POLICY = "loss"  # "loss" conservative


# =========================
# Utils
# =========================
def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def to_utc_datetime(ts: int) -> pd.Timestamp:
    # ts can be seconds or milliseconds
    if ts > 10_000_000_000:  # ms
        return pd.to_datetime(ts, unit="ms", utc=True)
    return pd.to_datetime(ts, unit="s", utc=True)


def format_pct_ptbr(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    s = f"{x:.{decimals}f}".replace(".", ",")
    return f"{s}%"


def mexc_request(path: str, params: Optional[dict] = None, timeout: int = 30) -> dict:
    url = MEXC_CONTRACT_BASE + path
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_tick_size_mexc_contract(symbol: str) -> float:
    """
    Best-effort tick size fetch from MEXC Contract detail endpoint.
    If not available, fallback to DEFAULT_TICK_SIZE.
    """
    try:
        j = mexc_request("/api/v1/contract/detail", params={"symbol": symbol})
        data = j.get("data", {}) if isinstance(j, dict) else {}
        # Common key in MEXC contract: "priceUnit"
        for k in ["priceUnit", "price_unit", "priceUnitPrecision", "price_unit_precision"]:
            if k in data:
                v = data[k]
                if isinstance(v, (int, float)) and float(v) > 0:
                    return float(v)
    except Exception:
        pass
    return float(DEFAULT_TICK_SIZE)


def _parse_kline_json(j: dict) -> pd.DataFrame:
    """
    Supports common MEXC contract kline response shapes:
    1) {"success":true,"data":{"time":[...],"open":[...],"high":[...],"low":[...],"close":[...],"vol":[...]}}
    2) {"success":true,"data":[{"time":...,"open":...,...}, ...]}
    """
    if not isinstance(j, dict):
        raise ValueError("Unexpected kline response type")

    data = j.get("data")
    if data is None:
        raise ValueError(f"No 'data' in response: keys={list(j.keys())}")

    # Array-of-fields format
    if isinstance(data, dict) and "time" in data:
        df = pd.DataFrame({
            "timestamp": data.get("time", []),
            "open": data.get("open", []),
            "high": data.get("high", []),
            "low": data.get("low", []),
            "close": data.get("close", []),
            "volume": data.get("vol", data.get("volume", [])),
        })
        return df

    # List-of-dicts format
    if isinstance(data, list):
        rows = []
        for r in data:
            if not isinstance(r, dict):
                continue
            rows.append({
                "timestamp": r.get("time", r.get("t")),
                "open": r.get("open", r.get("o")),
                "high": r.get("high", r.get("h")),
                "low": r.get("low", r.get("l")),
                "close": r.get("close", r.get("c")),
                "volume": r.get("vol", r.get("volume", r.get("v"))),
            })
        return pd.DataFrame(rows)

    raise ValueError("Unknown kline response schema")


def fetch_klines_mexc_contract_max_history(
    symbol: str,
    timeframe: str = "1h",
    limit_bars_per_call: int = 1000,
    pause_s: float = 0.15,
) -> pd.DataFrame:
    """
    Best-effort pagination backwards using start/end windows.
    Public endpoints can have limitations; we keep requesting older windows until no progress.
    """
    tf_map = {"1h": ("Min60", 60 * 60)}
    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe={timeframe} (this project uses only 1h).")

    interval, bar_seconds = tf_map[timeframe]

    end_ts = int(time.time())
    all_batches = []
    seen_min_ts = None

    for _ in range(10_000):  # safety cap
        start_ts = end_ts - limit_bars_per_call * bar_seconds
        params = {"interval": interval, "start": start_ts, "end": end_ts}

        j = mexc_request(f"/api/v1/contract/kline/{symbol}", params=params)
        df = _parse_kline_json(j)

        if df.empty or df["timestamp"].isna().all():
            break

        df = df.dropna(subset=["timestamp"]).copy()
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["timestamp"]).copy()
        df["timestamp"] = df["timestamp"].astype(np.int64)

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"]).copy()
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        batch_min = int(df["timestamp"].min())

        # stop if not moving backwards
        if seen_min_ts is not None and batch_min >= seen_min_ts:
            break
        seen_min_ts = batch_min

        all_batches.append(df)

        # move end backwards
        end_ts = batch_min - 1
        time.sleep(pause_s)

    if not all_batches:
        raise RuntimeError("Failed to fetch klines: API returned no usable data.")

    out = pd.concat(all_batches, ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    out["datetime"] = out["timestamp"].apply(to_utc_datetime)

    out = out[["timestamp", "datetime", "open", "high", "low", "close", "volume"]]
    return out


def load_or_fetch_data(symbol: str, timeframe: str = "1h") -> pd.DataFrame:
    ensure_dirs()
    path = f"data/{symbol}_{timeframe}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        else:
            df["datetime"] = df["timestamp"].apply(to_utc_datetime)

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
        df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
        return df

    df = fetch_klines_mexc_contract_max_history(symbol=symbol, timeframe=timeframe)
    df.to_csv(path, index=False)
    return df


# =========================
# Indicators / features
# =========================
def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr


def wilder_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df["close"].astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_new_high_context(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Features only (NO filtering):
    - bars_since_new_high
    - pullback_from_new_high_pct
    - after_new_high_flag
    and side-aware versions (same for long-only).
    """
    out = df.copy()
    highs = out["high"].astype(float).values
    closes = out["close"].astype(float).values

    bars_since = np.full(len(out), np.nan)
    pullback_pct = np.full(len(out), np.nan)
    after_flag = np.zeros(len(out), dtype=int)

    last_new_high = np.nan
    last_new_high_idx = None

    prev_roll_max = pd.Series(highs).shift(1).rolling(lookback, min_periods=lookback).max().values

    for i in range(len(out)):
        if i >= lookback and not np.isnan(prev_roll_max[i]):
            if highs[i] > prev_roll_max[i]:
                last_new_high = highs[i]
                last_new_high_idx = i

        if last_new_high_idx is None:
            after_flag[i] = 0
            bars_since[i] = np.nan
            pullback_pct[i] = np.nan
        else:
            after_flag[i] = 1
            bars_since[i] = i - last_new_high_idx
            if last_new_high and not np.isnan(last_new_high) and last_new_high != 0:
                pullback_pct[i] = (last_new_high - closes[i]) / last_new_high * 100.0
            else:
                pullback_pct[i] = np.nan

    out["bars_since_new_high"] = bars_since
    out["pullback_from_new_high_pct"] = pullback_pct
    out["after_new_high_flag"] = after_flag

    out["context_bars_since_extreme"] = out["bars_since_new_high"]
    out["context_pullback_pct"] = out["pullback_from_new_high_pct"]
    out["context_after_extreme_flag"] = out["after_new_high_flag"]

    return out


def add_indicators_and_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # SMA
    out["sma8"] = out["close"].rolling(8, min_periods=8).mean()
    out["sma80"] = out["close"].rolling(80, min_periods=80).mean()

    # slope helper: max SMA8 of previous 8 candles (strictly previous)
    out["sma8_prev8_max"] = out["sma8"].shift(1).rolling(8, min_periods=8).max()
    out["slope_up_flag"] = out["sma8"] > out["sma8_prev8_max"]

    out["ma_gap_pct"] = (out["sma8"] - out["sma80"]) / out["sma80"] * 100.0
    out["dist_to_sma80_pct"] = (out["close"] - out["sma80"]) / out["sma80"] * 100.0
    out["slope_strength"] = (out["sma8"] - out["sma8_prev8_max"]) / out["close"] * 100.0

    # ATR/RSI
    out["atr"] = wilder_atr(out, ATR_PERIOD)
    out["atr_pct"] = out["atr"] / out["close"] * 100.0
    out["rsi"] = wilder_rsi(out, RSI_PERIOD)

    # Candle anatomy
    out["range"] = out["high"] - out["low"]
    out["range_pct"] = out["range"] / out["close"] * 100.0
    out["body"] = (out["close"] - out["open"]).abs()
    out["body_pct"] = out["body"] / out["close"] * 100.0

    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["upper_wick_pct"] = out["upper_wick"] / out["close"] * 100.0
    out["lower_wick_pct"] = out["lower_wick"] / out["close"] * 100.0

    denom = (out["high"] - out["low"]).replace(0, np.nan)
    out["clv"] = (out["close"] - out["low"]) / denom

    # Momentum
    out["ret_1_pct"] = out["close"].pct_change(1) * 100.0
    out["ret_3_pct"] = out["close"].pct_change(3) * 100.0
    out["ret_5_pct"] = out["close"].pct_change(5) * 100.0

    # Structure N=20 (can include candle i, but no future)
    out["rolling_high_20"] = out["high"].rolling(STRUCT_N, min_periods=STRUCT_N).max()
    out["rolling_low_20"] = out["low"].rolling(STRUCT_N, min_periods=STRUCT_N).min()
    struct_denom = (out["rolling_high_20"] - out["rolling_low_20"]).replace(0, np.nan)

    out["pos_in_range_n"] = (out["close"] - out["rolling_low_20"]) / struct_denom
    out["dist_to_high_n_pct"] = (out["rolling_high_20"] - out["close"]) / out["close"] * 100.0
    out["dist_to_low_n_pct"] = (out["close"] - out["rolling_low_20"]) / out["close"] * 100.0

    # Volume zscore 20
    vol = out["volume"].astype(float)
    vol_mean = vol.rolling(20, min_periods=20).mean()
    vol_std = vol.rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
    out["vol_z"] = (vol - vol_mean) / vol_std

    # New high context (features only)
    out = add_new_high_context(out, lookback=EXTREME_LOOKBACK)

    return out


def extract_feature_row(df: pd.DataFrame, i: int) -> Dict:
    cols = [
        "sma8", "sma80",
        "ma_gap_pct", "dist_to_sma80_pct", "slope_strength",
        "atr", "atr_pct",
        "range", "range_pct",
        "body", "body_pct",
        "upper_wick", "lower_wick",
        "upper_wick_pct", "lower_wick_pct",
        "clv",
        "ret_1_pct", "ret_3_pct", "ret_5_pct",
        "rsi",
        "rolling_high_20", "rolling_low_20",
        "pos_in_range_n", "dist_to_high_n_pct", "dist_to_low_n_pct",
        "vol_z",
        "bars_since_new_high", "pullback_from_new_high_pct", "after_new_high_flag",
        "context_bars_since_extreme", "context_pullback_pct", "context_after_extreme_flag",
    ]
    row = {}
    for c in cols:
        row[c] = df.at[i, c] if c in df.columns else np.nan
    return row


# =========================
# Backtest core
# =========================
@dataclass
class TradeResult:
    entry_idx: int
    exit_idx: int
    outcome: str       # "win" / "loss"
    exit_reason: str   # "tp" / "sl"


def simulate_exit_long(df: pd.DataFrame, entry_idx: int, stop: float, tp: float) -> Optional[TradeResult]:
    """
    From entry_idx onwards, find first hit TP or SL.
    Conservative ambiguity: if candle hits both => LOSS (SL).
    """
    for j in range(entry_idx, len(df)):
        high = float(df.at[j, "high"])
        low = float(df.at[j, "low"])

        hit_tp = high >= tp
        hit_sl = low <= stop

        if hit_tp and hit_sl:
            if AMBIGUOUS_POLICY == "loss":
                return TradeResult(entry_idx=entry_idx, exit_idx=j, outcome="loss", exit_reason="sl")
            return TradeResult(entry_idx=entry_idx, exit_idx=j, outcome="win", exit_reason="tp")

        if hit_sl:
            return TradeResult(entry_idx=entry_idx, exit_idx=j, outcome="loss", exit_reason="sl")

        if hit_tp:
            return TradeResult(entry_idx=entry_idx, exit_idx=j, outcome="win", exit_reason="tp")

    return None


def backtest_one_rr(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    tick_size: float,
    rr: float,
    max_entry_wait_bars: int = 1,
) -> List[Dict]:
    """
    Runs a single RR simulation.
    Long-only, one position at a time.
    """
    trades: List[Dict] = []

    i = 2
    while i < len(df) - 2:
        # need indicators available + i+1 for entry check
        if pd.isna(df.at[i, "sma80"]) or pd.isna(df.at[i, "sma8_prev8_max"]):
            i += 1
            continue

        close_i = float(df.at[i, "close"])
        sma8_i = float(df.at[i, "sma8"])
        sma80_i = float(df.at[i, "sma80"])

        # Trend filter (long only)
        trend_ok = (close_i > sma8_i) and (close_i > sma80_i) and (sma8_i > sma80_i)
        # Slope filter (mandatory)
        slope_ok = bool(df.at[i, "slope_up_flag"])

        if not (trend_ok and slope_ok):
            i += 1
            continue

        # Setup candle i conditions
        low_i = float(df.at[i, "low"])
        low_1 = float(df.at[i - 1, "low"])
        low_2 = float(df.at[i - 2, "low"])
        high_i = float(df.at[i, "high"])
        close_1 = float(df.at[i - 1, "close"])

        pfr_long = (low_i < low_1) and (low_i < low_2) and (close_i > close_1)
        dl_long = (low_i < low_1) and (low_i < low_2)

        # If both trigger, prioritize PFR (more strict)
        setup = None
        if pfr_long:
            setup = "PFR"
        elif dl_long:
            setup = "DL"

        if setup is None:
            i += 1
            continue

        entry = high_i + tick_size
        stop = low_i - tick_size
        risk = entry - stop
        if risk <= 0:
            i += 1
            continue

        tp = entry + rr * risk

        # Entry wait: only next bar when max_entry_wait_bars=1 (spec)
        filled = False
        fill_idx = None
        for wait in range(1, max_entry_wait_bars + 1):
            j = i + wait
            if j >= len(df):
                break
            if float(df.at[j, "high"]) >= entry:
                filled = True
                fill_idx = j
                break

        if not filled:
            i += 1
            continue

        res = simulate_exit_long(df, entry_idx=fill_idx, stop=stop, tp=tp)
        if res is None:
            # no exit until end -> stop loop
            break

        feat = extract_feature_row(df, i)

        signal_time = df.at[i, "datetime"]
        entry_time = df.at[fill_idx, "datetime"]
        exit_time = df.at[res.exit_idx, "datetime"]

        trade = {
            "timeframe": timeframe,
            "symbol": symbol,
            "setup": setup,
            "side": "long",

            "signal_time": str(pd.Timestamp(signal_time).to_pydatetime().replace(tzinfo=None)) + "Z",
            "entry_time": str(pd.Timestamp(entry_time).to_pydatetime().replace(tzinfo=None)) + "Z",
            "exit_time": str(pd.Timestamp(exit_time).to_pydatetime().replace(tzinfo=None)) + "Z",

            "signal_idx": i,
            "entry_idx": fill_idx,
            "exit_idx": res.exit_idx,

            "entry": entry,
            "stop": stop,
            "risk": risk,
            "rr": rr,
            "tp": tp,

            "outcome": res.outcome,
            "exit_reason": res.exit_reason,

            "fill_delay": int(fill_idx - i),
            "bars_in_trade": int(res.exit_idx - fill_idx + 1),
        }
        trade.update(feat)
        trades.append(trade)

        # One position at a time: jump to next candle after exit
        i = res.exit_idx + 1

    return trades


def build_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["setup", "rr", "trades", "wins", "losses", "win_rate_pct"])

    rows = []
    for (setup, rr), grp in trades_df.groupby(["setup", "rr"], dropna=False):
        t = int(len(grp))
        w = int((grp["outcome"] == "win").sum())
        l = t - w
        wr = (w / t) * 100.0 if t > 0 else np.nan

        rows.append({
            "setup": setup,
            "rr": rr,
            "trades": t,
            "wins": w,
            "losses": l,
            "win_rate_pct": format_pct_ptbr(wr, decimals=1),
            "win_rate_pct_num": wr,
        })

    out = pd.DataFrame(rows).sort_values(["setup", "rr"]).reset_index(drop=True)
    return out
