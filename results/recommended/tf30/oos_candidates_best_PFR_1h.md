# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0350671 AND body_pct high60@0.178327 | 47 | 55,3% | 7,5% | 0.1064 | 0.1492 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 66 | 53,0% | 5,2% | 0.0606 | 0.1035 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.276009 | 57 | 52,6% | 4,8% | 0.0526 | 0.0955 |
| RR1_volz_high50 | 1 | vol_z high50@-0.280266 | 87 | 50,6% | 2,7% | 0.0115 | 0.0544 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31596 | 65 | 47,7% | -0,2% | -0.0462 | -0.0033 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.273797 | 54 | 48,1% | 7,4% | 0.2037 | 0.1852 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.29909 | 63 | 46,0% | 5,3% | 0.1508 | 0.1323 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 63 | 44,4% | 3,7% | 0.1111 | 0.0926 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.29227 | 87 | 42,5% | 1,8% | 0.0632 | 0.0447 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 75 | 45,3% | 9,5% | 0.3600 | 0.2837 |
| RR2_rsi_high60 | 1 | rsi high60@60.0456 | 80 | 45,0% | 9,1% | 0.3500 | 0.2737 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 62 | 40,3% | 4,4% | 0.2097 | 0.1333 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 81 | 39,5% | 3,6% | 0.1852 | 0.1088 |
| RR2_volz_high50 | 1 | vol_z high50@-0.310795 | 83 | 36,1% | 0,3% | 0.0843 | 0.0080 |

