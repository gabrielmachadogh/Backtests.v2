# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 62 | 56,5% | 8,3% | 0.1290 | 0.1666 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.279221 | 55 | 56,4% | 8,2% | 0.1273 | 0.1649 |
| RR1_volz_high50 | 1 | vol_z high50@-0.280266 | 79 | 54,4% | 6,3% | 0.0886 | 0.1262 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31273 | 69 | 47,8% | -0,3% | -0.0435 | -0.0059 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.277751 | 52 | 53,8% | 13,2% | 0.3462 | 0.3305 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 59 | 49,2% | 8,5% | 0.2288 | 0.2132 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24461 | 70 | 45,7% | 5,1% | 0.1429 | 0.1272 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.29227 | 79 | 45,6% | 4,9% | 0.1392 | 0.1236 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 57 | 43,9% | 9,2% | 0.3158 | 0.2755 |
| RR2_rsi_high60 | 1 | rsi high60@60.0876 | 76 | 43,4% | 8,7% | 0.3026 | 0.2623 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04786 | 67 | 43,3% | 8,6% | 0.2985 | 0.2582 |
| RR2_volz_high50 | 1 | vol_z high50@-0.32701 | 80 | 37,5% | 2,8% | 0.1250 | 0.0847 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 78 | 35,9% | 1,2% | 0.0769 | 0.0366 |

