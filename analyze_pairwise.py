import os
import itertools
import pandas as pd
import numpy as np
from tabulate import tabulate

SYMBOL = os.getenv("SYMBOL", "BTC_USDT")
INPUT_FILE = f"results/backtest_trades_{SYMBOL}.csv"
OUTPUT_FILE = f"results/pairwise_analysis_{SYMBOL}.csv"

N_BINS = 4          # 0 a 3
MIN_TRADES = 30     
MIN_WINRATE = 0.60  

def fmt_pct(val):
    try: return f"{val*100:.2f}%".replace(".", ",")
    except: return "-"

def analyze_pairwise():
    if not os.path.exists(INPUT_FILE): return
    df = pd.read_csv(INPUT_FILE)
    if df.empty: return

    # Identifica colunas
    rr_cols = [c for c in df.columns if c.startswith('rr_')]
    ignore = ['timeframe', 'setup'] + rr_cols
    features = [c for c in df.columns if c not in ignore]
    
    # Filtra features numéricas válidas
    valid_features = []
    for f in features:
        try:
            if df[f].nunique() > 1: valid_features.append(f)
        except: pass

    results = []
    print(f"Cruzando {len(valid_features)} variáveis...")

    for (tf, setup), group in df.groupby(['timeframe', 'setup']):
        # Binariza (0-3)
        bins_df = pd.DataFrame(index=group.index)
        for f in valid_features:
            try: bins_df[f] = pd.qcut(group[f], N_BINS, labels=False, duplicates='drop')
            except: pass
            
        feats_ready = bins_df.columns.tolist()

        for rr_col in rr_cols:
            rr_val = rr_col.replace('rr_', '')
            # 1=Win, 0=Loss
            outcomes = (group[rr_col] == 'win').astype(int)
            
            # Combinações 2 a 2
            for fa, fb in itertools.combinations(feats_ready, 2):
                # Agrupa
                tmp = pd.DataFrame({'res': outcomes, 'bin_a': bins_df[fa], 'bin_b': bins_df[fb]})
                stats = tmp.groupby(['bin_a', 'bin_b'])['res'].agg(['count', 'sum'])
                
                for (ba, bb), row in stats.iterrows():
                    trades = row['count']
                    wins = row['sum']
                    if trades < MIN_TRADES: continue
                    
                    wr = wins / trades
                    if wr >= MIN_WINRATE:
                        results.append({
                            'timeframe': tf, 'setup': setup, 'rr': rr_val,
                            'feature_a': fa, 'feature_b': fb,
                            'bin_a': int(ba), 'bin_b': int(bb),
                            'trades': int(trades), 'wins': int(wins),
                            'losses': int(trades - wins),
                            'win_rate': wr, 'win_rate_pct': fmt_pct(wr)
                        })

    if results:
        res_df = pd.DataFrame(results).sort_values('win_rate', ascending=False)
        res_df.to_csv(OUTPUT_FILE, index=False)
        
        # Salva MD
        with open(f"results/pairwise_best_{SYMBOL}.md", "w") as f:
            f.write(f"# Top Pairwise Combinations (>60% WR) - {SYMBOL}\n\n")
            f.write(tabulate(res_df.head(50), headers="keys", tablefmt="pipe", showindex=False))
            
        print("Análise concluída.")
    else:
        print("Nenhum padrão encontrado.")

if __name__ == "__main__":
    analyze_pairwise()
