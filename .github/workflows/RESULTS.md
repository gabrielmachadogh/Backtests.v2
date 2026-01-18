# Como ver os resultados do backtest no GitHub

Este repositório roda o backtest via **GitHub Actions** e salva os arquivos em uma branch separada chamada:

- `backtest-results`

A ideia é não “poluir” a branch `main` com CSVs grandes.

---

## 1) Rodar o backtest

1. Vá em **Actions**
2. Escolha o workflow: **Backtest BTC 1h Long (commit results)**
3. Clique em **Run workflow**
4. (Opcional) Ajuste inputs:
   - `symbol`: `BTC_USDT`
   - `timeframe`: `1h`
   - `rr_list`: `1.0,1.5,2.0,3.0`
5. Aguarde terminar (status verde)

---

## 2) Ver os resultados dentro do GitHub (sem baixar nada)

1. Vá em **Code**
2. Troque a branch de `main` para **`backtest-results`**
3. Abra a pasta `results/`

O arquivo mais fácil para começar é:

- `results/LATEST.md`

Ele contém:
- link do run do Actions
- links diretos para os CSVs/MDs
- tabela com winrates

---

## 3) Arquivos gerados (branch backtest-results)

Dentro de `results/` você vai encontrar:

- `backtest_trades_BTC_USDT_1h_long.csv`
  - 1 linha por trade, com todas as variáveis/features no candle de sinal
- `backtest_summary_BTC_USDT_1h_long.csv`
  - resumo por setup e RR (winrate)
- `patterns_univariate_BTC_USDT_1h_long.csv`
  - winrate por buckets (quartis + low/top percentis)
- `patterns_pairwise_BTC_USDT_1h_long.csv`
  - cruzamento de pares de features (bin_a/bin_b) e winrate
- `patterns_best_BTC_USDT_1h_long.md`
  - lista dos melhores padrões (filtrando por min_trades)
- `backtest_debug_BTC_USDT_1h_long.csv`
  - contagens internas (quantos candles passam filtros, quantos sinais, fills etc.)

---

## 4) Dica prática: como “usar” o patterns_best

1. Abra `results/patterns_best_BTC_USDT_1h_long.md`
2. Veja combinações com:
   - win_rate alto
   - trades >= 30 (ou seu threshold)
3. Depois, transforme o padrão escolhido em um filtro simples e re-rodar o backtest.
