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
    out["slope_strength"] = (out["sma8"] - out["sma8_prev8_max"]) / out["close"] * 100.0

    out["atr"] = wilder_atr(out, ATR_PERIOD)
    out["atr_pct"] = out["atr"] / out["close"] * 100.0
    out["rsi"] = wilder_rsi(out, RSI_PERIOD)

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

    out["ret_1_pct"] = out["close"].pct_change(1) * 100.0
    out["ret_3_pct"] = out["close"].pct_change(3) * 100.0
    out["ret_5_pct"] = out["close"].pct_change(5) * 100.0

    out["rolling_high_20"] = out["high"].rolling(STRUCT_N, min_periods=STRUCT_N).max()
    out["rolling_low_20"] = out["low"].rolling(STRUCT_N, min_periods=STRUCT_N).min()
    struct_denom = (out["rolling_high_20"] - out["rolling_low_20"]).replace(0, np.nan)

    out["pos_in_range_n"] = (out["close"] - out["rolling_low_20"]) / struct_denom
    out["dist_to_high_n_pct"] = (out["rolling_high_20"] - out["close"]) / out["close"] * 100.0
    out["dist_to_low_n_pct"] = (out["close"] - out["rolling_low_20"]) / out["close"] * 100.0

    vol = out["volume"].astype(float)
    vol_mean = vol.rolling(20, min_periods=20).mean()
    vol_std = vol.rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
    out["vol_z"] = (vol - vol_mean) / vol_std

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
    return {c: (df.at[i, c] if c in df.columns else np.nan) for c in cols}


# -------------------------
# backtest
# -------------------------
@dataclass
class TradeResult:
    entry_idx: int
    exit_idx: int
    outcome: str      # win/loss
    exit_reason: str  # tp/sl


def simulate_exit_long(df: pd.DataFrame, entry_idx: int, stop: float, tp: float) -> Optional[TradeResult]:
    for j in range(entry_idx, len(df)):
        high = float(df.at[j, "high"])
        low = float(df.at[j, "low"])

        hit_tp = high >= tp
        hit_sl = low <= stop

        if hit_tp and hit_sl:
            if AMBIGUOUS_POLICY == "loss":
                return TradeResult(entry_idx, j, "loss", "sl")
            return TradeResult(entry_idx, j, "win", "tp")

        if hit_sl:
            return TradeResult(entry_idx, j, "loss", "sl")
        if hit_tp:
            return TradeResult(entry_idx, j, "win", "tp")
    return None


def backtest_one_rr(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    tick_size: float,
    rr: float,
    max_entry_wait_bars: int = 1,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Returns (trades, stats) for debugging.
    """
    trades: List[Dict] = []
    stats = {
        "checked": 0,
        "trend_ok": 0,
        "slope_ok": 0,
        "both_ok": 0,
        "pfr_signals": 0,
        "dl_signals": 0,
        "chosen_pfr": 0,
        "chosen_dl": 0,
        "filled": 0,
        "exited": 0,
    }

    i = 2
    while i < len(df) - 2:
        stats["checked"] += 1

        if pd.isna(df.at[i, "sma80"]) or pd.isna(df.at[i, "sma8_prev8_max"]):
            i += 1
            continue

        close_i = float(df.at[i, "close"])
        sma8_i = float(df.at[i, "sma8"])
        sma80_i = float(df.at[i, "sma80"])

        trend_ok = (close_i > sma8_i) and (close_i > sma80_i) and (sma8_i > sma80_i)
        if trend_ok:
            stats["trend_ok"] += 1

        slope_ok = bool(df.at[i, "slope_up_flag"])
        if slope_ok:
            stats["slope_ok"] += 1

        if not (trend_ok and slope_ok):
            i += 1
            continue

        stats["both_ok"] += 1

        low_i = float(df.at[i, "low"])
        low_1 = float(df.at[i - 1, "low"])
        low_2 = float(df.at[i - 2, "low"])
        high_i = float(df.at[i, "high"])
        close_1 = float(df.at[i - 1, "close"])

        pfr_long = (low_i < low_1) and (low_i < low_2) and (close_i > close_1)
        dl_long = (low_i < low_1) and (low_i < low_2)

        if pfr_long:
            stats["pfr_signals"] += 1
        if dl_long:
            stats["dl_signals"] += 1

        setup = "PFR" if pfr_long else ("DL" if dl_long else None)
        if setup is None:
            i += 1
            continue

        if setup == "PFR":
            stats["chosen_pfr"] += 1
        else:
            stats["chosen_dl"] += 1

        entry = high_i + tick_size
        stop = low_i - tick_size
        risk = entry - stop
        if risk <= 0:
            i += 1
            continue

        tp = entry + rr * risk

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

        stats["filled"] += 1

        res = simulate_exit_long(df, entry_idx=fill_idx, stop=stop, tp=tp)
        if res is None:
            break

        stats["exited"] += 1

        feat = extract_feature_row(df, i)

        trade = {
            "timeframe": timeframe,
            "symbol": symbol,
            "setup": setup,
            "side": "long",
            "signal_time": df.at[i, "datetime"].isoformat(),
            "entry_time": df.at[fill_idx, "datetime"].isoformat(),
            "exit_time": df.at[res.exit_idx, "datetime"].isoformat(),
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

        i = res.exit_idx + 1  # one trade at a time

    return trades, stats


def build_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["setup", "rr", "trades", "wins", "losses", "win_rate_pct", "win_rate_pct_num"])

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
            "win_rate_pct": format_pct_ptbr(wr, 1),
            "win_rate_pct_num": wr,
        })
    return pd.DataFrame(rows).sort_values(["setup", "rr"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTC_USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--only_long", type=int, default=1)
    parser.add_argument("--max_entry_wait", type=int, default=1)
    parser.add_argument("--tick_size", type=float, default=None)
    parser.add_argument("--rr_list", type=str, default="1.0,1.5,2.0,3.0")
    args = parser.parse_args()

    symbol = args.symbol
    timeframe = args.timeframe
    max_entry_wait = int(args.max_entry_wait)
    rr_list = [float(x.strip()) for x in args.rr_list.split(",") if x.strip()]

    ensure_dirs()

    df = load_or_fetch_data(symbol=symbol, timeframe=timeframe)
    df = add_indicators_and_features(df)

    # --- tick size
    if args.tick_size is not None:
        tick_size = float(args.tick_size)
        tick_raw = None
        tick_source = "cli"
    else:
        tick_size, tick_raw = get_tick_size_mexc_contract(symbol)
        tick_source = "mexc"

    # --- diagnostics about data
    if len(df) > 0:
        start_dt = df["datetime"].iloc[0]
        end_dt = df["datetime"].iloc[-1]
    else:
        start_dt = None
        end_dt = None

    print("=== DATA DIAGNOSTICS ===")
    print(f"symbol={symbol} timeframe={timeframe}")
    print(f"candles={len(df)} start={start_dt} end={end_dt}")
    print(f"sma80_available={int(df['sma80'].notna().sum())}")
    print(f"slope_available={int(df['sma8_prev8_max'].notna().sum())}")
    print("=== TICK ===")
    print(f"tick_source={tick_source} tick_raw={tick_raw} tick_size_used={tick_size}")
    print("========================")

    all_trades: List[Dict] = []
    debug_rows = []

    for rr in rr_list:
        trades_rr, stats = backtest_one_rr(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            tick_size=tick_size,
            rr=rr,
            max_entry_wait_bars=max_entry_wait,
        )
        all_trades.extend(trades_rr)

        stats_row = {"rr": rr}
        stats_row.update(stats)
        debug_rows.append(stats_row)

        print(f"=== RR {rr} STATS ===")
        for k in ["checked", "trend_ok", "slope_ok", "both_ok", "pfr_signals", "dl_signals", "chosen_pfr", "chosen_dl", "filled", "exited"]:
            print(f"{k}={stats.get(k)}")
        print(f"trades={len(trades_rr)}")
        print("=====================")

    # garante header mesmo com 0 trades
    base_cols = [
        "timeframe", "symbol", "setup", "side",
        "signal_time", "entry_time", "exit_time",
        "signal_idx", "entry_idx", "exit_idx",
        "entry", "stop", "risk", "rr", "tp",
        "outcome", "exit_reason",
        "fill_delay", "bars_in_trade",
        "sma8", "sma80", "ma_gap_pct", "dist_to_sma80_pct", "slope_strength",
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
    trades_df = pd.DataFrame(all_trades, columns=base_cols)

    # paths (novo + compat)
    out_trades_new = f"results/backtest_trades_{symbol}_{timeframe}_long.csv"
    out_trades_compat = f"results/backtest_trades_{symbol}.csv"
    trades_df.to_csv(out_trades_new, index=False)
    trades_df.to_csv(out_trades_compat, index=False)

    summary_df = build_summary(trades_df)
    out_summary_new = f"results/backtest_summary_{symbol}_{timeframe}_long.csv"
    out_summary_compat = f"results/backtest_summary_{symbol}.csv"
    summary_df.to_csv(out_summary_new, index=False)
    summary_df.to_csv(out_summary_compat, index=False)

    # debug stats CSV
    debug_df = pd.DataFrame(debug_rows)
    debug_path = f"results/backtest_debug_{symbol}_{timeframe}_long.csv"
    debug_df.to_csv(debug_path, index=False)

    print(f"Saved trades (new):     {out_trades_new}")
    print(f"Saved trades (compat):  {out_trades_compat}")
    print(f"Saved summary (new):    {out_summary_new}")
    print(f"Saved summary (compat): {out_summary_compat}")
    print(f"Saved debug:            {debug_path}")
    print(f"Trades total: {len(trades_df)}")


if __name__ == "__main__":
    main()
