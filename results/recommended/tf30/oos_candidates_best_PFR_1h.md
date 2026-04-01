# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.28011 | 57 | 59,6% | 10,4% | 0.1930 | 0.2084 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 64 | 59,4% | 10,1% | 0.1875 | 0.2029 |
| RR1_volz_high50 | 1 | vol_z high50@-0.291583 | 78 | 57,7% | 8,5% | 0.1538 | 0.1692 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31399 | 65 | 47,7% | -1,5% | -0.0462 | -0.0308 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.278776 | 54 | 55,6% | 15,6% | 0.3889 | 0.3889 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 61 | 50,8% | 10,8% | 0.2705 | 0.2705 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24529 | 65 | 46,2% | 6,2% | 0.1538 | 0.1538 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.306383 | 76 | 46,1% | 6,1% | 0.1513 | 0.1513 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 59 | 42,4% | 8,5% | 0.2712 | 0.2547 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 69 | 42,0% | 8,1% | 0.2609 | 0.2443 |
| RR2_rsi_high60 | 1 | rsi high60@60.0961 | 72 | 40,3% | 6,4% | 0.2083 | 0.1918 |
| RR2_volz_high50 | 1 | vol_z high50@-0.333782 | 76 | 38,2% | 4,3% | 0.1447 | 0.1282 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 76 | 34,2% | 0,3% | 0.0263 | 0.0098 |

