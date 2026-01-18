#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


MEXC_CONTRACT_BASE = "https://contract.mexc.com"
DEFAULT_TICK_SIZE = 0.1

RR_LIST_DEFAULT = [1.0, 1.5, 2.0, 3.0]
ATR_PERIOD = 14
RSI_PERIOD = 14
STRUCT_N = 20
EXTREME_LOOKBACK = 20

# Conservative policy: if TP and SL hit in same candle => loss
AMBIGUOUS_POLICY = "loss"


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


def safe_div(a, b):
    return np.where(b == 0, np.nan, a / b)


def mexc_request(path: str, params: Optional[dict] = None, timeout: int = 30) -> dict:
    url = MEXC_CONTRACT_BASE + path
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_tick_size_mexc_contract(symbol: str) -> float:
    """
    Try to fetch tick size (price unit) for contract symbol from MEXC.
    Fallback to DEFAULT_TICK_SIZE if not found.
    """
    try:
        j = mexc_request("/api/v1/contract/detail", params={"symbol": symbol})
        data = j.get("data", {}) if isinstance(j, dict) else {}
        # different possible keys across versions
        for k in ["priceUnit", "price_unit", "price_unit_precision", "priceUnitPrecision"]:
            if k in data:
                v = data[k]
                if isinstance(v, (int, float)) and v > 0:
                    return float(v)
        # sometimes nested
        if "contractSize" in data and "priceUnit" in data:
            v = data["priceUnit"]
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    except Exception:
        pass
    return float(DEFAULT_TICK_SIZE)


def _parse_kline_json(j: dict) -> pd.DataFrame:
    """
    Supports common MEXC contract kline response shapes:
    - {"success":true,"data":{"time":[...],"open":[...],"high":[...],"low":[...],"close":[...],"vol":[...]}}
    - {"success":true,"data":[{"time":...,"open":...,...}, ...]}
    """
    if not isinstance(j, dict):
        raise ValueError("Unexpected kline response type")

    data = j.get("data")
    if data is None:
        raise ValueError(f"No 'data' in response: keys={list(j.keys())}")

    # Array-of-fields format
    if isinstance(data, dict) and "time" in data:
        time_arr = data.get("time", [])
        open_arr = data.get("open", [])
        high_arr = data.get("high", [])
        low_arr = data.get("low", [])
        close_arr = data.get("close", [])
        vol_arr = data.get("vol", data.get("volume", []))

        df = pd.DataFrame({
            "timestamp": time_arr,
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
            "volume": vol_arr,
        })
        return df

    # List-of-dicts format
    if isinstance(data, list):
        # try multiple key variants
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
        df = pd.DataFrame(rows)
        return df

    raise ValueError("Unknown kline response schema")


def fetch_klines_mexc_contract_max_history(
    symbol: str,
    timeframe: str = "1h",
    limit_bars_per_call: int = 1000,
    pause_s: float = 0.15,
) -> pd.DataFrame:
    """
    Paginate backwards using start/end windows. This is a best-effort "max history"
    approach given public API limitations.
    """
    tf_map = {"1h": "Min60"}  # extend if needed
    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe={timeframe} (only 1h in this project)")

    interval = tf_map[timeframe]
    bar_seconds = 60 * 60

    end_ts = int(time.time())
    all_batches = []
    seen_min_ts = None

    # loop backward until API returns empty / no progress
    for _ in range(10_000):  # hard safety
        start_ts = end_ts - limit_bars_per_call * bar_seconds
        params = {
            "interval": interval,
            "start": start_ts,
            "end": end_ts,
        }
        j = mexc_request(f"/api/v1/contract/kline/{symbol}", params=params)
        df = _parse_kline_json(j)

        if df.empty or df["timestamp"].isna().all():
            break

        df = df.dropna(subset=["timestamp"]).copy()
        df["timestamp"] = df["timestamp"].astype(np.int64)

        # Ensure numeric
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"]).copy()
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        batch_min = int(df["timestamp"].min())
        batch_max = int(df["timestamp"].max())

        # Detect no progress
        if seen_min_ts is not None and batch_min >= seen_min_ts:
            break
        seen_min_ts = batch_min

        all_batches.append(df)

        # move end backwards
        end_ts = batch_min - 1

        time.sleep(pause_s)

    if not all_batches:
        raise RuntimeError("Failed to fetch klines (no data returned).")

    out = pd.concat(all_batches, ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    # Normalize to UTC datetime
    out["datetime"] = out["timestamp"].apply(to_utc_datetime)
    out = out.sort_values("datetime").reset_index(drop=True)

    # Keep canonical columns order
    out = out[["timestamp", "datetime", "open", "high", "low", "close", "volume"]]
    return out


def load_or_fetch_data(symbol: str, timeframe: str = "1h") -> pd.DataFrame:
    ensure_dirs()
    path = f"data/{symbol}_{timeframe}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Parse datetime as UTC
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        else:
            df["datetime"] = df["timestamp"].apply(to_utc_datetime)
        df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        return df

    df = fetch_klines_mexc_contract_max_history(symbol=symbol, timeframe=timeframe)
    df.to_csv(path, index=False)
    return df


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


def add_indicators_and_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sma8"] = out["close"].rolling(8, min_periods=8).mean()
    out["sma80"] = out["close"].rolling(80, min_periods=80).mean()

    # slope filter helper: max SMA8 of previous 8 candles (strictly previous)
    out["sma8_prev8_max"] = out["sma8"].shift(1).rolling(8, min_periods=8).max()
    out["slope_up_flag"] = out["sma8"] > out["sma8_prev8_max"]

    out["ma_gap_pct"] = (out["sma8"] - out["sma80"]) / out["sma80"] * 100.0
    out["dist_to_sma80_pct"] = (out["close"] - out["sma80"]) / out["sma80"] * 100.0

    # slope_strength per spec
    out["slope_strength"] = (out["sma8"] - out["sma8_prev8_max"]) / out["close"] * 100.0

    # ATR / RSI
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

    # Momentum returns
    out["ret_1_pct"] = out["close"].pct_change(1) * 100.0
    out["ret_3_pct"] = out["close"].pct_change(3) * 100.0
    out["ret_5_pct"] = out["close"].pct_change(5) * 100.0

    # Structure (N=20)
    out["rolling_high_20"] = out["high"].rolling(STRUCT_N, min_periods=STRUCT_N).max()
    out["rolling_low_20"] = out["low"].rolling(STRUCT_N, min_periods=STRUCT_N).min()

    struct_denom = (out["rolling_high_20"] - out["rolling_low_20"]).replace(0, np.nan)
    out["pos_in_range_n"] = (out["close"] - out["rolling_low_20"]) / struct_denom
    out["dist_to_high_n_pct"] = (out["rolling_high_20"] - out["close"]) / out["close"] * 100.0
    out["dist_to_low_n_pct"] = (out["close"] - out["rolling_low_20"]) / out["close"] * 100.0

    # Volume zscore (20)
    vol = out["volume"].astype(float)
    vol_mean = vol.rolling(20, min_periods=20).mean()
    vol_std = vol.rolling(20, min_periods=20).std(ddof=0)
    out["vol_z"] = (vol - vol_mean) / vol_std.replace(0, np.nan)

    # Extreme context: "new high + pullback" (features only)
    out = add_new_high_context(out, lookback=EXTREME_LOOKBACK)

    return out


def add_new_high_context(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    out = df.copy()
    highs = out["high"].astype(float).values
    closes = out["close"].astype(float).values

    bars_since = np.full(len(out), np.nan)
    pullback_pct = np.full(len(out), np.nan)
    after_flag = np.zeros(len(out), dtype=int)

    last_new_high = np.nan
    last_new_high_idx = None

    # Precompute rolling max of previous lookback highs (strictly previous)
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

    # side-aware versions (only long => same)
    out["context_bars_since_extreme"] = out["bars_since_new_high"]
    out["context_pullback_pct"] = out["pullback_from_new_high_pct"]
    out["context_after_extreme_flag"] = out["after_new_high_flag"]

    return out


@dataclass
class TradeResult:
    entry_idx: int
    exit_idx: int
    outcome: str  # "win" or "loss"
    exit_reason: str  # "tp" or "sl"


def simulate_exit_long(
    df: pd.DataFrame,
    entry_idx: int,
    stop: float,
    tp: float,
) -> Optional[TradeResult]:
    """
    Starting at entry_idx candle, find first hit of SL or TP.
    Conservative on ambiguity (same candle hits both => loss).
    """
    for j in range(entry_idx, len(df)):
        high = float(df.at[j, "high"])
        low = float(df.at[j, "low"])

        hit_tp = high >= tp
        hit_sl = low <= stop

        if hit_tp and hit_sl:
            # ambiguous
            if AMBIGUOUS_POLICY == "loss":
                return TradeResult(entry_idx=entry_idx, exit_idx=j, outcome="loss", exit_reason="sl")
            else:
                return TradeResult(entry_idx=entry_idx, exit_idx=j, outcome="win", exit_reason="tp")

        if hit_sl:
            return TradeResult(entry_idx=entry_idx, exit_idx=j, outcome="loss", exit_reason="sl")

        if hit_tp:
            return TradeResult(entry_idx=entry_idx, exit_idx=j, outcome="win", exit_reason="tp")

    return None


def extract_feature_row(df: pd.DataFrame, i: int) -> Dict:
    """
    Features are taken from the SIGNAL candle i (no lookahead).
    """
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


def backtest_one_rr(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    tick_size: float,
    rr: float,
    max_entry_wait_bars: int = 1,
    only_long: bool = True,
) -> List[Dict]:
    """
    Run a single-RR backtest enforcing "one trade at a time".
    Only long is supported here by design.
    """
    if not only_long:
        raise ValueError("This backtest is configured for only_long=True")

    trades: List[Dict] = []

    # Minimum index to have i-2, indicators, slope window etc.
    # Also need i+1 for entry check.
    i = 2
    while i < len(df) - 2:
        # Skip if indicators/features are not available
        if pd.isna(df.at[i, "sma80"]) or pd.isna(df.at[i, "sma8_prev8_max"]):
            i += 1
            continue

        close = float(df.at[i, "close"])
        sma8 = float(df.at[i, "sma8"])
        sma80 = float(df.at[i, "sma80"])

        # Regime (trend up) for longs
        trend_ok = (close > sma8) and (close > sma80) and (sma8 > sma80)
        slope_ok = bool(df.at[i, "slope_up_flag"])

        if not (trend_ok and slope_ok):
            i += 1
            continue

        # Setup conditions at SIGNAL candle i
        low_i = float(df.at[i, "low"])
        low_1 = float(df.at[i - 1, "low"])
        low_2 = float(df.at[i - 2, "low"])

        high_i = float(df.at[i, "high"])
        close_1 = float(df.at[i - 1, "close"])

        pfr_long = (low_i < low_1) and (low_i < low_2) and (close > close_1)
        dl_long = (low_i < low_1) and (low_i < low_2)

        # Evaluate each setup independently, but still "one position at a time":
        # if both trigger, prioritize PFR (more strict) then DL (you can change if desired).
        setup_to_take = None
        if pfr_long:
            setup_to_take = "PFR"
        elif dl_long:
            setup_to_take = "DL"

        if setup_to_take is None:
            i += 1
            continue

        entry = high_i + tick_size
        stop = low_i - tick_size
        risk = entry - stop
        if not (risk > 0):
            i += 1
            continue

        tp = entry + rr * risk

        # Entry rule: MAX_ENTRY_WAIT_BARS=1 => only next candle i+1
        # (kept generic but fixed to 1 by spec)
        filled = False
        fill_idx = None

        # We'll implement generic loop but expects max_entry_wait_bars=1
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

        # Simulate exit from entry candle onward
        res = simulate_exit_long(df, entry_idx=fill_idx, stop=stop, tp=tp)
        if res is None:
            # No exit hit until end; discard or mark as open. We'll discard by default.
            break

        signal_time = df.at[i, "datetime"]
        entry_time = df.at[fill_idx, "datetime"]
        exit_time = df.at[res.exit_idx, "datetime"]

        feat = extract_feature_row(df, i)

        trade = {
            "timeframe": timeframe,
            "symbol": symbol,
            "setup": setup_to_take,
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
            "fill_delay": (fill_idx - i),
            "bars_in_trade": (res.exit_idx - fill_idx + 1),
        }
        trade.update(feat)
        trades.append(trade)

        # Enforce "one trade at a time": jump i to exit index + 1
        i = res.exit_idx + 1

    return trades


def build_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["setup", "rr", "trades", "wins", "losses", "win_rate_pct"])

    g = trades_df.groupby(["setup", "rr"], dropna=False)
    rows = []
    for (setup, rr), grp in g:
        trades = int(len(grp))
        wins = int((grp["outcome"] == "win").sum())
        losses = trades - wins
        win_rate = (wins / trades) * 100.0 if trades > 0 else np.nan
        rows.append({
            "setup": setup,
            "rr": rr,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": format_pct_ptbr(win_rate, decimals=1),
            "win_rate_pct_num": win_rate,
        })

    out = pd.DataFrame(rows).sort_values(["setup", "rr"]).reset_index(drop=True)
    return out


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
    only_long = bool(args.only_long)
    max_entry_wait = int(args.max_entry_wait)
    rr_list = [float(x.strip()) for x in args.rr_list.split(",") if x.strip()]

    ensure_dirs()

    df = load_or_fetch_data(symbol=symbol, timeframe=timeframe)
    df = add_indicators_and_features(df)

    tick_size = args.tick_size
    if tick_size is None:
        tick_size = get_tick_size_mexc_contract(symbol)

    all_trades = []
    for rr in rr_list:
        trades_rr = backtest_one_rr(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            tick_size=tick_size,
            rr=rr,
            max_entry_wait_bars=max_entry_wait,
            only_long=only_long,
        )
        all_trades.extend(trades_rr)

    trades_df = pd.DataFrame(all_trades)
    out_trades_path = f"results/backtest_trades_{symbol}_{timeframe}_long.csv"
    trades_df.to_csv(out_trades_path, index=False)

    summary_df = build_summary(trades_df)
    out_summary_path = f"results/backtest_summary_{symbol}_{timeframe}_long.csv"
    # keep both numeric and formatted; you can drop numeric later if you want
    summary_df.to_csv(out_summary_path, index=False)

    print(f"Saved trades:  {out_trades_path} ({len(trades_df)} rows)")
    print(f"Saved summary: {out_summary_path} ({len(summary_df)} rows)")
    print(f"Tick size used: {tick_size}")


if __name__ == "__main__":
    main()def parse_kline(payload):
    # Proteção 1: Payload inválido
    if not payload or not isinstance(payload, dict): return pd.DataFrame()
    
    data = payload.get("data", []) or payload.get("result", [])
    
    # Proteção 2: Lista vazia
    if not data or not isinstance(data, list) or len(data) == 0: 
        return pd.DataFrame()
    
    cols = ['ts', 'open', 'high', 'low', 'close', 'vol']
    
    try:
        # Formato lista de listas [ts, o, h, l, c, v]
        if isinstance(data[0], list): 
            df = pd.DataFrame(data, columns=['time'] + cols[1:])
        # Formato lista de dicts
        elif isinstance(data[0], dict):
            df = pd.DataFrame(data)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Erro parseando: {e}")
        return pd.DataFrame()
    
    # Padronizar
    if 'time' in df.columns: df = df.rename(columns={'time': 'ts'})
    if 'vol' in df.columns: df = df.rename(columns={'vol': 'volume'})
    
    required = ['ts', 'open', 'high', 'low', 'close', 'volume']
    for c in required:
        if c not in df.columns: df[c] = np.nan
    
    df = df[required].copy()
    
    # Numérico
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    # Data
    if not df['ts'].empty:
        # Detecção automática ms ou s
        unit = 'ms' if df['ts'].iloc[0] > 1e11 else 's'
        df['ts'] = pd.to_datetime(pd.to_numeric(df['ts']), unit=unit, utc=True)
    
    return df.sort_values('ts').reset_index(drop=True)

def fetch_history(symbol):
    print(f"Baixando histórico 1h para {symbol}...")
    all_dfs = []
    end_ts = int(time.time())
    step = WINDOW_DAYS * 86400
    
    # Loop de segurança
    for _ in range(50): # Max 50 calls
        if len(all_dfs) * 24 * WINDOW_DAYS >= MAX_BARS_FETCH: break
        
        start_ts = end_ts - step
        # Tenta pegar dados
        data = http_get_json(f"{BASE_URL}/contract/kline/{symbol}", {'interval': 'Min60', 'start': start_ts, 'end': end_ts})
        
        df = parse_kline(data)
        
        if df.empty: break
        
        all_dfs.append(df)
        
        # Atualiza timestamp para o próximo bloco (andando para trás)
        first_ts = df.iloc[0]['ts'].timestamp()
        if first_ts >= end_ts: break # Evita loop infinito se API devolver mesmo bloco
        end_ts = int(first_ts) - 1
        
        time.sleep(0.2)
        
    if not all_dfs: return pd.DataFrame()
    full_df = pd.concat(all_dfs).drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    return full_df.tail(MAX_BARS_FETCH).reset_index(drop=True)

def resample(df, rule):
    if df.empty: return df
    if rule == '1h': return df.copy()
    mapping = {'2h': '2h', '4h': '4h', '1d': '1d'}
    df = df.set_index('ts')
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    res = df.resample(mapping[rule]).agg(agg).dropna()
    return res.reset_index()

# --- LÓGICA & FEATURES ---
def add_features(df):
    x = df.copy()
    x['sma_s'] = x['close'].rolling(SMA_SHORT).mean()
    x['sma_l'] = x['close'].rolling(SMA_LONG).mean()
    
    # 1. Trend & Slope
    x['trend_up'] = (x['close'] > x['sma_s']) & (x['close'] > x['sma_l']) & (x['sma_s'] > x['sma_l'])
    x['slope_up'] = x['sma_s'] > x['sma_s'].shift(1)
    
    # Feature: Força da inclinação
    x['slope_strength'] = x['sma_s'] - x['sma_s'].shift(SLOPE_LOOKBACK)

    # 2. Contexto de Topo
    roll_hi = x['high'].rolling(20).max()
    x['is_new_high'] = x['high'] >= roll_hi
    
    x['grp_hi'] = x['is_new_high'].cumsum()
    x['bars_since_new_high'] = x.groupby('grp_hi').cumcount()
    
    x['last_high_price'] = x['high'].where(x['is_new_high']).ffill()
    x['pullback_from_new_high_pct'] = (x['last_high_price'] - x['close']) / x['last_high_price'] * 100
    
    # 3. Contexto de Fundo
    roll_lo = x['low'].rolling(20).min()
    x['dist_to_low_n_pct'] = (x['close'] - roll_lo) / x['close'] * 100
    x['dist_to_high_n_pct'] = (roll_hi - x['close']) / x['close'] * 100
    
    # 4. Anatomia do Candle
    x['range'] = x['high'] - x['low']
    x['body'] = abs(x['close'] - x['open'])
    x['upper_wick'] = x['high'] - np.maximum(x['open'], x['close'])
    x['lower_wick'] = np.minimum(x['open'], x['close']) - x['low']
    
    # Evita divisão por zero
    mask = x['range'] > 0
    x['body_pct'] = np.where(mask, (x['body'] / x['range']) * 100, 0)
    x['upper_wick_pct'] = np.where(mask, (x['upper_wick'] / x['range']) * 100, 0)
    x['lower_wick_pct'] = np.where(mask, (x['lower_wick'] / x['range']) * 100, 0)
    x['clv'] = np.where(mask, (x['close'] - x['low']) / x['range'], 0.5)
    
    # 5. Momentum e Volatilidade
    x['ret_1_pct'] = x['close'].pct_change(1) * 100
    x['ret_3_pct'] = x['close'].pct_change(3) * 100
    x['ret_5_pct'] = x['close'].pct_change(5) * 100
    
    x['atr'] = (x['high'] - x['low']).rolling(14).mean()
    x['atr_pct'] = (x['atr'] / x['close']) * 100
    x['range_pct'] = (x['range'] / x['close']) * 100
    
    x['dist_to_sma80_pct'] = ((x['close'] - x['sma_l']) / x['sma_l']) * 100
    x['ma_gap_pct'] = ((x['sma_s'] - x['sma_l']) / x['sma_l']) * 100
    
    x['vol_z'] = (x['volume'] - x['volume'].rolling(20).mean()) / x['volume'].rolling(20).std()
    
    delta = x['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    x['rsi'] = 100 - (100 / (1 + rs))
    
    rng = roll_hi - roll_lo
    x['pos_in_range_n'] = np.where(rng > 0, (x['close'] - roll_lo) / rng, 0.5)
    
    x['context_after_extreme_flag'] = (x['bars_since_new_high'] <= 5).astype(int)
    x['context_pullback_pct'] = x['pullback_from_new_high_pct'] 
    x['context_bars_since_extreme'] = x['bars_since_new_high']

    return x

def check_signals(x, i):
    # PFR Long
    pfr = (x['low'].iloc[i] < x['low'].iloc[i-1]) and \
          (x['low'].iloc[i] < x['low'].iloc[i-2]) and \
          (x['close'].iloc[i] > x['close'].iloc[i-1])
          
    # DL Long
    dl = (x['low'].iloc[i] < x['low'].iloc[i-1]) and \
         (x['low'].iloc[i] < x['low'].iloc[i-2])
         
    return pfr, dl

def run_backtest(df, tf):
    x = add_features(df)
    if x.empty: return pd.DataFrame()
    
    tick = df['close'].iloc[-1] * 0.0001
    trades = []
    
    start_idx = SMA_LONG + 20
    
    # Features para salvar
    feature_cols = [
        'slope_strength', 'bars_since_new_high', 'pullback_from_new_high_pct',
        'dist_to_sma80_pct', 'atr_pct', 'clv', 'lower_wick_pct', 'upper_wick_pct',
        'body_pct', 'range_pct', 'pos_in_range_n', 'ret_1_pct', 'ret_3_pct', 'ret_5_pct', 
        'vol_z', 'rsi', 'ma_gap_pct', 'dist_to_high_n_pct', 'dist_to_low_n_pct',
        'context_pullback_pct', 'context_bars_since_extreme', 'context_after_extreme_flag'
    ]
    
    for i in range(start_idx, len(x) - 10):
        if not (x['trend_up'].iloc[i] and x['slope_up'].iloc[i]): continue
            
        pfr, dl = check_signals(x, i)
        active = []
        if pfr: active.append('PFR')
        if dl and not pfr: active.append('DL')
        
        if not active: continue
        
        for setup in active:
            entry = x['high'].iloc[i] + tick
            stop = x['low'].iloc[i] - tick
            
            if x['high'].iloc[i+1] < entry: continue # Não ativou
            
            fill_idx = i + 1
            res = {'timeframe': tf, 'setup': setup}
            for feat in feature_cols:
                res[feat] = x[feat].iloc[i]
            
            for rr in RRS:
                target = entry + (abs(entry-stop) * rr)
                outcome = 'loss'
                for k in range(fill_idx, min(fill_idx+50, len(x))):
                    if x['low'].iloc[k] <= stop: break
                    if x['high'].iloc[k] >= target: 
                        outcome = 'win'
                        break
                res[f"rr_{rr}"] = outcome
            
            trades.append(res)
            
    return pd.DataFrame(trades)

def main():
    os.makedirs("results", exist_ok=True)
    df_raw = fetch_history(SYMBOL)
    
    if df_raw.empty:
        # Cria vazio
        pd.DataFrame(columns=['timeframe', 'setup']).to_csv(f"results/backtest_trades_{SYMBOL}.csv", index=False)
        return

    all_trades = []
    for tf in TIMEFRAMES:
        try:
            df_tf = resample(df_raw, tf)
            t = run_backtest(df_tf, tf)
            if not t.empty: all_trades.append(t)
        except: pass

    if all_trades:
        final = pd.concat(all_trades)
        final.to_csv(f"results/backtest_trades_{SYMBOL}.csv", index=False)
        print(f"Trades gerados: {len(final)}")
    else:
        pd.DataFrame(columns=['timeframe', 'setup']).to_csv(f"results/backtest_trades_{SYMBOL}.csv", index=False)
        print("0 trades.")

if __name__ == "__main__":
    main()
