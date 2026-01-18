#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest BTC_USDT perp (MEXC) - 1h - ONLY LONG - Setups: PFR + Dave Landry

Regras:
- Trend (no candle do sinal i): close > SMA8, close > SMA80, SMA8 > SMA80
- Slope obrigatório (no candle i): SMA8[i] > max(SMA8[i-8:i]) (8 anteriores)
- Setups (candle i):
  PFR long: low[i] < low[i-1] and low[i] < low[i-2] and close[i] > close[i-1]
  DL  long: low[i] < low[i-1] and low[i] < low[i-2]
  Entrada: buy stop em high[i] + 1 tick
- Execução: MAX_ENTRY_WAIT_BARS=1 (só pode preencher no candle i+1)
- Stop: low[i] - 1 tick
- TP: entry + RR * risco, RR=[1.0,1.5,2.0,3.0]
- 1 trade por vez
- Sem custos
- Ambíguo (TP e SL no mesmo candle): LOSS

Outputs (ambos):
- results/backtest_trades_{symbol}_{timeframe}_long.csv  (novo)
- results/backtest_trades_{symbol}.csv                  (compat)
- results/backtest_summary_{symbol}_{timeframe}_long.csv (novo)
- results/backtest_summary_{symbol}.csv                  (compat)
"""

import os
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


MEXC_CONTRACT_BASE = "https://contract.mexc.com"
DEFAULT_TICK_SIZE = 0.1

ATR_PERIOD = 14
RSI_PERIOD = 14
STRUCT_N = 20
EXTREME_LOOKBACK = 20

AMBIGUOUS_POLICY = "loss"  # conservative


# -------------------------
# utils
# -------------------------
def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def to_utc_datetime(ts: int) -> pd.Timestamp:
    if ts > 10_000_000_000:  # ms
        return pd.to_datetime(ts, unit="ms", utc=True)
    return pd.to_datetime(ts, unit="s", utc=True)


def format_pct_ptbr(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{x:.{decimals}f}".replace(".", ",") + "%"


def mexc_request(path: str, params: Optional[dict] = None, timeout: int = 30) -> dict:
    url = MEXC_CONTRACT_BASE + path
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def normalize_tick_size(raw: Optional[float]) -> float:
    """
    Heurística:
    - Se vier um inteiro pequeno (ex.: 1,2,3...) pode ser 'precisão' em casas decimais.
      Ex.: 1 => 0.1; 2 => 0.01; 3 => 0.001 ...
    - Se vier float < 1, assume que já é tick.
    """
    if raw is None:
        return float(DEFAULT_TICK_SIZE)

    try:
        v = float(raw)
    except Exception:
        return float(DEFAULT_TICK_SIZE)

    if not np.isfinite(v) or v <= 0:
        return float(DEFAULT_TICK_SIZE)

    # provável "precision" (1..8) em vez de tick
    if v >= 1 and v <= 8 and float(v).is_integer():
        return float(10 ** (-int(v)))

    return float(v)


def get_tick_size_mexc_contract(symbol: str) -> Tuple[float, Optional[float]]:
    """
    Best-effort tick size fetch from MEXC Contract detail endpoint.
    Returns: (tick_size_normalized, raw_value_if_found)
    """
    raw_val = None
    try:
        j = mexc_request("/api/v1/contract/detail", params={"symbol": symbol})
        data = j.get("data", {}) if isinstance(j, dict) else {}
        for k in ["priceUnit", "price_unit", "priceUnitPrecision", "price_unit_precision"]:
            if k in data:
                v = data[k]
                if isinstance(v, (int, float)) and float(v) > 0:
                    raw_val = float(v)
                    break
    except Exception:
        raw_val = None

    tick = normalize_tick_size(raw_val)
    return tick, raw_val


def _parse_kline_json(j: dict) -> pd.DataFrame:
    data = j.get("data")
    if data is None:
        raise ValueError(f"No 'data' in response: keys={list(j.keys())}")

    if isinstance(data, dict) and "time" in data:
        return pd.DataFrame({
            "timestamp": data.get("time", []),
            "open": data.get("open", []),
            "high": data.get("high", []),
            "low": data.get("low", []),
            "close": data.get("close", []),
            "volume": data.get("vol", data.get("volume", [])),
        })

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
    tf_map = {"1h": ("Min60", 60 * 60)}
    if timeframe not in tf_map:
        raise ValueError("Only 1h supported in this project.")

    interval, bar_seconds = tf_map[timeframe]

    end_ts = int(time.time())
    batches = []
    seen_min_ts = None

    for _ in range(10_000):
        start_ts = end_ts - limit_bars_per_call * bar_seconds
        params = {"interval": interval, "start": start_ts, "end": end_ts}

        j = mexc_request(f"/api/v1/contract/kline/{symbol}", params=params)
        df = _parse_kline_json(j)

        if df.empty or df["timestamp"].isna().all():
            break

        df = df.dropna(subset=["timestamp"]).copy()
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()
        df["timestamp"] = df["timestamp"].astype(np.int64)

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"]).copy()
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        batch_min = int(df["timestamp"].min())
        if seen_min_ts is not None and batch_min >= seen_min_ts:
            break
        seen_min_ts = batch_min

        batches.append(df)
        end_ts = batch_min - 1
        time.sleep(pause_s)

    if not batches:
        raise RuntimeError("Failed to fetch klines (no data).")

    out = pd.concat(batches, ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    out["datetime"] = out["timestamp"].apply(to_utc_datetime)
    return out[["timestamp", "datetime", "open", "high", "low", "close", "volume"]]


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


# -------------------------
# indicators/features
# -------------------------
def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def wilder_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df["close"].astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_new_high_context(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
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
        else:
            after_flag[i] = 1
            bars_since[i] = i - last_new_high_idx
            if last_new_high and not np.isnan(last_new_high) and last_new_high != 0:
                pullback_pct[i] = (last_new_high - closes[i]) / last_new_high * 100.0

    out["bars_since_new_high"] = bars_since
    out["pullback_from_new_high_pct"] = pullback_pct
    out["after_new_high_flag"] = after_flag

    out["context_bars_since_extreme"] = out["bars_since_new_high"]
    out["context_pullback_pct"] = out["pullback_from_new_high_pct"]
    out["context_after_extreme_flag"] = out["after_new_high_flag"]
    return out


def add_indicators_and_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sma8"] = out["close"].rolling(8, min_periods=8).mean()
    out["sma80"] = out["close"].rolling(80, min_periods=80).mean()

    out["sma8_prev8_max"] = out["sma8"].shift(1).rolling(8, min_periods=8).max()
    out["slope_up_flag"] = out["sma8"] > out["sma8_prev8_max"]

    out["ma_gap_pct"] = (out["sma8"] - out["sma80"]) / out["sma80"] * 100.0
    out["dist_to_sma80_pct"] = (out["close"] - out["sma80"]) / out["sma80"] * 100.0
    out["slope_strength"] = (out["sma8"] - 
