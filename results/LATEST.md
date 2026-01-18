# LATEST (PFR 1h | RR 1.0/1.5/2.0)

- Repo: `gabrielmachadogh/Backtests.v2`
- Branch de resultados: `backtest-results`
- Workflow run: https://github.com/gabrielmachadogh/Backtests.v2/actions/runs/21116845086
- UTC: 2026-01-18 18:52:00

## Arquivos (backtest)
- Trades: https://github.com/gabrielmachadogh/Backtests.v2/blob/backtest-results/results/backtest_trades_BTC_USDT_1h_long.csv
- Summary: https://github.com/gabrielmachadogh/Backtests.v2/blob/backtest-results/results/backtest_summary_BTC_USDT_1h_long.csv
- Debug: https://github.com/gabrielmachadogh/Backtests.v2/blob/backtest-results/results/backtest_debug_BTC_USDT_1h_long.csv

## Arquivos (OOS: sem overfitting)
- OOS best (.md): https://github.com/gabrielmachadogh/Backtests.v2/blob/backtest-results/results/oos_best_PFR_1h.md
- OOS baseline (.csv): https://github.com/gabrielmachadogh/Backtests.v2/blob/backtest-results/results/oos_baseline_PFR_1h.csv
- OOS univariate (.csv): https://github.com/gabrielmachadogh/Backtests.v2/blob/backtest-results/results/oos_univariate_PFR_1h.csv
- OOS pairwise (.csv): https://github.com/gabrielmachadogh/Backtests.v2/blob/backtest-results/results/oos_pairwise_PFR_1h.csv
- OOS inconclusive (.csv): https://github.com/gabrielmachadogh/Backtests.v2/blob/backtest-results/results/oos_inconclusive_features_PFR_1h.csv

## Como usar
1) Abra o **OOS best** e veja os filtros com melhor **delta no TESTE**.
2) Se uma variável cair em **inconclusive**, ignore ela.
3) Compare regras antigas vs novas de topo pelos features:
   - antigo: after_new_high_flag / context_after_extreme_flag
   - novo: after_new_high_recent_flag / context_after_extreme_flag_v2 / pullback_from_new_high_atr

