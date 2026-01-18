name: Backtest BTC 1h Long (PFR OOS + Candidates)

on:
  workflow_dispatch:
    inputs:
      symbol:
        description: "Símbolo (MEXC perp), ex: BTC_USDT"
        required: true
        default: "BTC_USDT"
      timeframe:
        description: "Timeframe"
        required: true
        default: "1h"
      min_trades_test:
        description: "Min trades no teste (OOS) para aceitar filtro"
        required: true
        default: "40"
      min_trades_train:
        description: "Min trades no treino (OOS) para aceitar filtro"
        required: true
        default: "120"
      test_frac:
        description: "Fração do histórico usada como teste (ex: 0.2)"
        required: true
        default: "0.2"
      min_improvement_pp:
        description: "Melhora mínima (pontos percentuais) no TESTE para considerar variável útil"
        required: true
        default: "1.0"
      results_branch:
        description: "Branch para salvar results/"
        required: true
        default: "backtest-results"

  schedule:
    - cron: "0 3 * * *"

concurrency:
  group: backtest-results
  cancel-in-progress: true

permissions:
  contents: write

jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    env:
      SYMBOL: ${{ inputs.symbol || 'BTC_USDT' }}
      TIMEFRAME: ${{ inputs.timeframe || '1h' }}
      RESULTS_BRANCH: ${{ inputs.results_branch || 'backtest-results' }}

      TEST_FRAC: ${{ inputs.test_frac || '0.2' }}
      MIN_TRADES_TEST: ${{ inputs.min_trades_test || '40' }}
      MIN_TRADES_TRAIN: ${{ inputs.min_trades_train || '120' }}
      MIN_IMPROVEMENT_PP: ${{ inputs.min_improvement_pp || '1.0' }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        shell: bash
        run: |
          set -euo pipefail
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then
            pip install -r requirements.txt
          else
            pip install pandas numpy requests
          fi

      - name: Run backtest (PFR only)
        shell: bash
        working-directory: ${{ github.workspace }}
        run: |
          set -euo pipefail
          python backtest.py \
            --symbol "$SYMBOL" \
            --timeframe "$TIMEFRAME" \
            --only_long 1 \
            --max_entry_wait 1 \
            --tick_size 0.1 \
            --rr_list "1.0,1.5,2.0"

      - name: Run OOS analysis (univariate + pairwise + redundant)
        shell: bash
        working-directory: ${{ github.workspace }}
        run: |
          set -euo pipefail
          TRADES_FILE="results/backtest_trades_${SYMBOL}_${TIMEFRAME}_long.csv"
          if [ ! -f "$TRADES_FILE" ]; then
            echo "ERRO: não encontrei $TRADES_FILE"
            ls -la results || true
            exit 1
          fi

          python analyze_patterns.py \
            --trades "$TRADES_FILE" \
            --setup "PFR" \
            --timeframe "$TIMEFRAME" \
            --test_frac "$TEST_FRAC" \
            --min_trades_train "$MIN_TRADES_TRAIN" \
            --min_trades_test "$MIN_TRADES_TEST" \
            --min_improvement_pp "$MIN_IMPROVEMENT_PP" \
            --max_pairwise_features 20 \
            --percentiles "10,15,20,25,30,40,50,60"

      - name: Run OOS Candidates (fixed rules)
        shell: bash
        working-directory: ${{ github.workspace }}
        run: |
          set -euo pipefail
          TRADES_FILE="results/backtest_trades_${SYMBOL}_${TIMEFRAME}_long.csv"
          python evaluate_candidates.py \
            --trades "$TRADES_FILE" \
            --setup "PFR" \
            --timeframe "$TIMEFRAME" \
            --test_frac "$TEST_FRAC" \
            --min_trades_train "$MIN_TRADES_TRAIN" \
            --min_trades_test "$MIN_TRADES_TEST"

      - name: Generate results/LATEST.md
        shell: bash
        working-directory: ${{ github.workspace }}
        run: |
          set -euo pipefail

          REPO="${GITHUB_REPOSITORY}"
          RUN_URL="https://github.com/${REPO}/actions/runs/${GITHUB_RUN_ID}"
          BRANCH="${RESULTS_BRANCH}"

          TRADES="results/backtest_trades_${SYMBOL}_${TIMEFRAME}_long.csv"
          SUMMARY="results/backtest_summary_${SYMBOL}_${TIMEFRAME}_long.csv"
          DEBUG="results/backtest_debug_${SYMBOL}_${TIMEFRAME}_long.csv"

          OOS_BEST="results/oos_best_PFR_1h.md"
          OOS_BASE="results/oos_baseline_PFR_1h.csv"
          OOS_UNI="results/oos_univariate_PFR_1h.csv"
          OOS_PAIR="results/oos_pairwise_PFR_1h.csv"
          OOS_INCONC="results/oos_inconclusive_features_PFR_1h.csv"
          OOS_REDUND="results/oos_redundant_features_PFR_1h.csv"

          CAND_CSV="results/oos_candidates_report_PFR_1h.csv"
          CAND_MD="results/oos_candidates_best_PFR_1h.md"

          link() { echo "https://github.com/${REPO}/blob/${BRANCH}/$1"; }

          {
            echo "# LATEST (PFR 1h | RR 1.0/1.5/2.0)"
            echo
            echo "- Repo: \`${REPO}\`"
            echo "- Branch de resultados: \`${BRANCH}\`"
            echo "- Workflow run: ${RUN_URL}"
            echo "- UTC: $(date -u '+%Y-%m-%d %H:%M:%S')"
            echo
            echo "## Backtest"
            echo "- Trades: $(link "${TRADES}")"
            echo "- Summary: $(link "${SUMMARY}")"
            echo "- Debug: $(link "${DEBUG}")"
            echo
            echo "## OOS (sem overfitting: thresholds no treino)"
            echo "- OOS best (.md): $(link "${OOS_BEST}")"
            echo "- OOS baseline (.csv): $(link "${OOS_BASE}")"
            echo "- OOS univariate (.csv): $(link "${OOS_UNI}")"
            echo "- OOS pairwise (.csv): $(link "${OOS_PAIR}")"
            echo "- OOS inconclusive (.csv): $(link "${OOS_INCONC}")"
            echo "- OOS redundant (features duplicadas removidas do pairwise): $(link "${OOS_REDUND}")"
            echo
            echo "## Candidatos operacionais (regras FIXAS, sem search)"
            echo "- Candidates best (.md): $(link "${CAND_MD}")"
            echo "- Candidates report (.csv): $(link "${CAND_CSV}")"
            echo
            echo "## Como usar"
            echo "1) Leia **Candidates best**: ele te diz quais regras fixas melhoram OOS (e quanto)."
            echo "2) Leia **OOS best** para ideias adicionais, mas evite 'caçar o melhor' demais."
            echo "3) Compare topo antigo vs novo:"
            echo "   - antigo: after_new_high_flag / context_after_extreme_flag"
            echo "   - novo: after_new_high_recent_flag / context_after_extreme_flag_v2 / pullback_from_new_high_atr"
            echo
          } > results/LATEST.md

      - name: Commit results/ to results branch (checkout-safe)
        shell: bash
        working-directory: ${{ github.workspace }}
        run: |
          set -euo pipefail

          rm -rf /tmp/results_snapshot
          mkdir -p /tmp/results_snapshot
          cp -r results /tmp/results_snapshot/results

          git reset --hard
          git clean -fd

          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

          git fetch origin "$RESULTS_BRANCH" || true

          if git show-ref --verify --quiet "refs/remotes/origin/${RESULTS_BRANCH}"; then
            git checkout -B "$RESULTS_BRANCH" "origin/${RESULTS_BRANCH}"
          else
            git checkout -B "$RESULTS_BRANCH"
          fi

          rm -rf results
          cp -r /tmp/results_snapshot/results results

          git add results/
          if git diff --cached --quiet; then
            echo "No changes in results/. Nothing to commit."
            exit 0
          fi

          git commit -m "PFR ${SYMBOL} ${TIMEFRAME} OOS + Candidates (RR=1/1.5/2) [run ${GITHUB_RUN_ID}]"
          git push origin "$RESULTS_BRANCH"
