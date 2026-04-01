# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.275492 | 52 | 50,0% | 6,3% | 0.0000 | 0.1264 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 43 | 48,8% | 5,2% | -0.0233 | 0.1032 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.270164 | 36 | 47,2% | 3,5% | -0.0556 | 0.0709 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24094 | 46 | 43,5% | -0,2% | -0.1304 | -0.0040 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.269444 | 35 | 45,7% | 9,6% | 0.1429 | 0.2392 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.21828 | 45 | 42,2% | 6,1% | 0.0556 | 0.1519 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 42 | 40,5% | 4,3% | 0.0119 | 0.1083 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.283478 | 50 | 40,0% | 3,9% | 0.0000 | 0.0964 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8238 AND pullback_from_new_high_atr low50@1.05693 | 39 | 35,9% | 5,0% | 0.0769 | 0.1510 |
| RR2_rsi_high60 | 1 | rsi high60@59.8238 | 54 | 35,2% | 4,3% | 0.0556 | 0.1296 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 49 | 34,7% | 3,8% | 0.0408 | 0.1149 |
| RR2_volz_high50 | 1 | vol_z high50@-0.29227 | 49 | 34,7% | 3,8% | 0.0408 | 0.1149 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 41 | 34,1% | 3,3% | 0.0244 | 0.0985 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 53 | 32,1% | 1,2% | -0.0377 | 0.0363 |

