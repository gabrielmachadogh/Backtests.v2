# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.249369 | 57 | 49,1% | 2,9% | -0.0175 | 0.0577 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 41 | 48,8% | 2,5% | -0.0244 | 0.0509 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.270919 | 37 | 48,6% | 2,4% | -0.0270 | 0.0482 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23978 | 46 | 47,8% | 1,6% | -0.0435 | 0.0318 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22274 | 45 | 48,9% | 8,9% | 0.2222 | 0.2222 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.270667 | 36 | 44,4% | 4,4% | 0.1111 | 0.1111 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 40 | 42,5% | 2,5% | 0.0625 | 0.0625 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.277879 | 60 | 40,0% | 0,0% | 0.0000 | 0.0000 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@60.0298 AND pullback_from_new_high_atr low50@1.04658 | 38 | 52,6% | 17,0% | 0.5789 | 0.5100 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04658 | 50 | 46,0% | 10,4% | 0.3800 | 0.3110 |
| RR2_rsi_high60 | 1 | rsi high60@60.0298 | 54 | 44,4% | 8,8% | 0.3333 | 0.2644 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 39 | 38,5% | 2,8% | 0.1538 | 0.0849 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 55 | 36,4% | 0,7% | 0.0909 | 0.0219 |
| RR2_volz_high50 | 1 | vol_z high50@-0.286689 | 57 | 35,1% | -0,5% | 0.0526 | -0.0163 |

