import os
import time
import requests
import numpy as np
import pandas as pd

# Removemos dependência não usada para evitar erro se esquecer requirements
# from tabulate import tabulate 

BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")
SYMBOL = os.getenv("SYMBOL", "BTC_USDT")

TIMEFRAMES = ["1h", "2h", "4h", "1d"]
SETUPS = ["PFR", "DL"]

# Parâmetros
SMA_SHORT = 8
SMA_LONG = 80
SLOPE_LOOKBACK = 8
MAX_ENTRY_WAIT_BARS = 1
MAX_HOLD_BARS = 50
RRS = [1.0, 1.5, 2.0]

# Download
MAX_BARS_FETCH = 20000 
WINDOW_DAYS = 30 

# --- DADOS ---
def http_get_json(url, params=None, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200: return r.json()
        except: time.sleep(1)
    return None

def parse_kline(payload):
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
