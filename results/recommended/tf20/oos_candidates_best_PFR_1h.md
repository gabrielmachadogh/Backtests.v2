# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.239186 | 55 | 49,1% | 4,5% | -0.0182 | 0.0905 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 40 | 47,5% | 2,9% | -0.0500 | 0.0587 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.270919 | 37 | 45,9% | 1,4% | -0.0811 | 0.0276 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24461 | 44 | 45,5% | 0,9% | -0.0909 | 0.0178 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23498 | 42 | 45,2% | 7,0% | 0.1310 | 0.1759 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.270667 | 36 | 44,4% | 6,2% | 0.1111 | 0.1561 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 39 | 41,0% | 2,8% | 0.0256 | 0.0706 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.275492 | 58 | 39,7% | 1,5% | -0.0086 | 0.0363 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9577 AND pullback_from_new_high_atr low50@1.04786 | 37 | 51,4% | 18,8% | 0.5405 | 0.5638 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04786 | 48 | 43,8% | 11,2% | 0.3125 | 0.3358 |
| RR2_rsi_high60 | 1 | rsi high60@59.9577 | 55 | 41,8% | 9,3% | 0.2545 | 0.2778 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 37 | 35,1% | 2,6% | 0.0541 | 0.0773 |
| RR2_volz_high50 | 1 | vol_z high50@-0.288793 | 56 | 33,9% | 1,4% | 0.0179 | 0.0411 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 55 | 32,7% | 0,2% | -0.0182 | 0.0051 |

