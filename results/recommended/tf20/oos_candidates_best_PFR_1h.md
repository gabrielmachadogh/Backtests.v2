# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.267522 | 51 | 51,0% | 7,3% | 0.0196 | 0.1460 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 43 | 48,8% | 5,2% | -0.0233 | 0.1032 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.269913 | 36 | 47,2% | 3,5% | -0.0556 | 0.0709 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23748 | 47 | 42,6% | -1,1% | -0.1489 | -0.0225 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.269444 | 35 | 45,7% | 10,0% | 0.1429 | 0.2500 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.21828 | 46 | 41,3% | 5,6% | 0.0326 | 0.1398 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 42 | 40,5% | 4,8% | 0.0119 | 0.1190 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.283478 | 51 | 39,2% | 3,5% | -0.0196 | 0.0875 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.7534 AND pullback_from_new_high_atr low50@1.05743 | 40 | 37,5% | 6,6% | 0.1250 | 0.1991 |
| RR2_rsi_high60 | 1 | rsi high60@59.7534 | 56 | 35,7% | 4,9% | 0.0714 | 0.1455 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 40 | 35,0% | 4,1% | 0.0500 | 0.1241 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 49 | 34,7% | 3,8% | 0.0408 | 0.1149 |
| RR2_volz_high50 | 1 | vol_z high50@-0.291583 | 49 | 34,7% | 3,8% | 0.0408 | 0.1149 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 54 | 31,5% | 0,6% | -0.0556 | 0.0185 |

