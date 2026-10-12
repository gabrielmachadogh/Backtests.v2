# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23748 | 46 | 45,7% | 2,0% | -0.0870 | 0.0407 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 43 | 44,2% | 0,6% | -0.1163 | 0.0114 |
| RR1_volz_high50 | 1 | vol_z high50@-0.234643 | 57 | 43,9% | 0,2% | -0.1228 | 0.0049 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.269913 | 39 | 43,6% | -0,0% | -0.1282 | -0.0005 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.21828 | 45 | 46,7% | 8,2% | 0.1667 | 0.2051 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.269444 | 38 | 39,5% | 1,0% | -0.0132 | 0.0253 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 42 | 38,1% | -0,4% | -0.0476 | -0.0092 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.249369 | 56 | 37,5% | -1,0% | -0.0625 | -0.0240 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9895 AND pullback_from_new_high_atr low50@1.04486 | 40 | 47,5% | 13,4% | 0.4250 | 0.4023 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04486 | 50 | 42,0% | 7,9% | 0.2600 | 0.2373 |
| RR2_rsi_high60 | 1 | rsi high60@59.9895 | 58 | 41,4% | 7,3% | 0.2414 | 0.2187 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 54 | 35,2% | 1,1% | 0.0556 | 0.0328 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 40 | 35,0% | 0,9% | 0.0500 | 0.0273 |
| RR2_volz_high50 | 1 | vol_z high50@-0.285748 | 58 | 31,0% | -3,1% | -0.0690 | -0.0917 |

