# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 41 | 48,8% | 4,3% | -0.0244 | 0.0867 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.271242 | 37 | 45,9% | 1,5% | -0.0811 | 0.0300 |
| RR1_volz_high50 | 1 | vol_z high50@-0.259552 | 57 | 45,6% | 1,2% | -0.0877 | 0.0234 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.25253 | 45 | 44,4% | 0,0% | -0.1111 | 0.0000 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.271206 | 36 | 44,4% | 7,7% | 0.1111 | 0.1916 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24094 | 44 | 40,9% | 4,1% | 0.0227 | 0.1032 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 40 | 40,0% | 3,2% | 0.0000 | 0.0805 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.280266 | 57 | 35,1% | -1,7% | -0.1228 | -0.0423 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8816 AND pullback_from_new_high_atr low50@1.05792 | 39 | 46,2% | 14,0% | 0.3846 | 0.4203 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05792 | 51 | 41,2% | 9,0% | 0.2353 | 0.2710 |
| RR2_rsi_high60 | 1 | rsi high60@59.8816 | 57 | 38,6% | 6,5% | 0.1579 | 0.1936 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 37 | 35,1% | 3,0% | 0.0541 | 0.0898 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 55 | 32,7% | 0,6% | -0.0182 | 0.0175 |
| RR2_volz_high50 | 1 | vol_z high50@-0.29227 | 56 | 32,1% | 0,0% | -0.0357 | 0.0000 |

