# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.279665 | 57 | 59,6% | 10,4% | 0.1930 | 0.2084 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 64 | 59,4% | 10,1% | 0.1875 | 0.2029 |
| RR1_volz_high50 | 1 | vol_z high50@-0.290896 | 78 | 57,7% | 8,5% | 0.1538 | 0.1692 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31578 | 64 | 46,9% | -2,4% | -0.0625 | -0.0471 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.278331 | 54 | 55,6% | 15,6% | 0.3889 | 0.3889 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 61 | 50,8% | 10,8% | 0.2705 | 0.2705 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24667 | 65 | 46,2% | 6,2% | 0.1538 | 0.1538 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.293428 | 76 | 46,1% | 6,1% | 0.1513 | 0.1513 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 58 | 43,1% | 9,2% | 0.2931 | 0.2766 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 68 | 42,6% | 8,8% | 0.2794 | 0.2629 |
| RR2_rsi_high60 | 1 | rsi high60@60.1055 | 72 | 40,3% | 6,4% | 0.2083 | 0.1918 |
| RR2_volz_high50 | 1 | vol_z high50@-0.338748 | 77 | 37,7% | 3,8% | 0.1299 | 0.1133 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 77 | 33,8% | -0,1% | 0.0130 | -0.0035 |

