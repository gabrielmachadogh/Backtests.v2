# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.279665 | 57 | 59,6% | 10,0% | 0.1930 | 0.2006 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 64 | 59,4% | 9,8% | 0.1875 | 0.1951 |
| RR1_volz_high50 | 1 | vol_z high50@-0.290896 | 78 | 57,7% | 8,1% | 0.1538 | 0.1615 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31578 | 65 | 47,7% | -1,9% | -0.0462 | -0.0385 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.278331 | 54 | 55,6% | 15,1% | 0.3889 | 0.3770 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 61 | 50,8% | 10,3% | 0.2705 | 0.2586 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24667 | 66 | 47,0% | 6,5% | 0.1742 | 0.1623 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.293428 | 76 | 46,1% | 5,6% | 0.1513 | 0.1394 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 58 | 43,1% | 8,7% | 0.2931 | 0.2603 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 68 | 42,6% | 8,2% | 0.2794 | 0.2466 |
| RR2_rsi_high60 | 1 | rsi high60@60.1055 | 73 | 41,1% | 6,7% | 0.2329 | 0.2001 |
| RR2_volz_high50 | 1 | vol_z high50@-0.338748 | 77 | 37,7% | 3,2% | 0.1299 | 0.0971 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 78 | 34,6% | 0,2% | 0.0385 | 0.0057 |

