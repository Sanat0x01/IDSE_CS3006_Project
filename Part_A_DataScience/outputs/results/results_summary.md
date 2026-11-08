# Auto-generated results summary

_Written by `run_pipeline.py`; every number in REPORT.md comes from a run like this one._


## Data audit
```json
{
 "rows_in": 487680,
 "series_in": 3810,
 "dtypes_after": {
  "sensor": "float32",
  "ids": "int32",
  "surface": "str"
 },
 "duplicate_rows_exact": 0,
 "duplicate_keys": 0,
 "duplicate_label_rows": 0,
 "missing_sensor_values": 0,
 "missing_labels": 0,
 "series_dropped_missing": 0,
 "series_dropped_incomplete": 0,
 "duplicate_series": 0,
 "quaternion_norm_min": 0.9999922513961792,
 "quaternion_norm_max": 1.0000075101852417,
 "quaternions_renormalised": 0,
 "outlier_pct_robust_z_gt_6": {
  "orientation_X": 0.0,
  "orientation_Y": 0.0,
  "orientation_Z": 0.0,
  "orientation_W": 0.0,
  "angular_velocity_X": 1.867,
  "angular_velocity_Y": 0.153,
  "angular_velocity_Z": 2.313,
  "linear_acceleration_X": 1.582,
  "linear_acceleration_Y": 0.566,
  "linear_acceleration_Z": 3.031
 },
 "groups_with_multiple_surfaces": 0,
 "series_out": 3810,
 "groups": 73,
 "rows_out": 487680
}
```

## Split (by session)
| index | windows | sessions | Firm windows | Compliant windows | Irregular windows | Firm sessions | Compliant sessions | Irregular sessions |
|---|---|---|---|---|---|---|---|---|
| train | 2311 | 43 | 1374 | 731 | 206 | 26 | 14 | 3 |
| val | 743 | 15 | 455 | 243 | 45 | 9 | 5 | 1 |
| test | 756 | 15 | 455 | 244 | 57 | 9 | 5 | 1 |

## Best hyper-parameters (grouped CV on TRAIN)
```json
{
 "LogReg": {
  "clf__C": 0.1
 },
 "RandomForest": {
  "n_estimators": 200,
  "min_samples_leaf": 3,
  "max_features": 0.2,
  "max_depth": 8
 },
 "CNN": {
  "width": 16,
  "dropout": 0.3,
  "lr": 0.003
 }
}
```

### CV results - LogReg (top 5)
| params | mean_test_f1_macro | std_test_f1_macro | mean_test_hazard_recall | rank_test_f1_macro |
|---|---|---|---|---|
| {'clf__C': 0.1} | 0.669 | 0.046 | 0.780 | 1 |
| {'clf__C': 1.0} | 0.662 | 0.037 | 0.765 | 2 |
| {'clf__C': 10.0} | 0.651 | 0.040 | 0.743 | 3 |
| {'clf__C': 0.01} | 0.640 | 0.055 | 0.809 | 4 |
| {'clf__C': 0.001} | 0.637 | 0.082 | 0.851 | 5 |

### CV results - RandomForest (top 5)
| params | mean_test_f1_macro | std_test_f1_macro | mean_test_hazard_recall | rank_test_f1_macro |
|---|---|---|---|---|
| {'n_estimators': 200, 'min_samples_leaf': 3, 'max_features': 0.2, 'max_depth': 8} | 0.696 | 0.073 | 0.728 | 1 |
| {'n_estimators': 200, 'min_samples_leaf': 8, 'max_features': 'sqrt', 'max_depth': None} | 0.688 | 0.075 | 0.727 | 2 |
| {'n_estimators': 200, 'min_samples_leaf': 8, 'max_features': 'sqrt', 'max_depth': 8} | 0.687 | 0.089 | 0.761 | 3 |
| {'n_estimators': 200, 'min_samples_leaf': 3, 'max_features': 0.4, 'max_depth': None} | 0.687 | 0.054 | 0.659 | 4 |
| {'n_estimators': 200, 'min_samples_leaf': 8, 'max_features': 0.2, 'max_depth': None} | 0.686 | 0.064 | 0.710 | 5 |

### CV results - CNN (top 5)
| params | mean_test_f1_macro | std_test_f1_macro | mean_test_hazard_recall | rank_test_f1_macro |
|---|---|---|---|---|
| {'width': 16, 'dropout': 0.3, 'lr': 0.003} | 0.673 | 0.015 | 0.703 | 1 |
| {'width': 16, 'dropout': 0.5, 'lr': 0.003} | 0.672 | 0.039 | 0.708 | 2 |
| {'width': 32, 'dropout': 0.3, 'lr': 0.003} | 0.665 | 0.023 | 0.637 | 3 |
| {'width': 16, 'dropout': 0.3, 'lr': 0.001} | 0.665 | 0.033 | 0.734 | 4 |
| {'width': 32, 'dropout': 0.5, 'lr': 0.001} | 0.665 | 0.045 | 0.694 | 5 |

## VAL metrics - rule: argmax
| model | accuracy | balanced_accuracy | macro_f1 | hazard_recall | hazard_precision | hazard_f2 | miss_rate | false_alarm_rate | severe_miss_irregular_as_firm |
|---|---|---|---|---|---|---|---|---|---|
| Majority | 0.612 | 0.333 | 0.253 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| LogReg | 0.651 | 0.489 | 0.469 | 0.774 | 0.578 | 0.725 | 0.226 | 0.358 | 0.356 |
| RandomForest | 0.715 | 0.521 | 0.493 | 0.799 | 0.639 | 0.761 | 0.201 | 0.286 | 0.467 |
| CNN | 0.704 | 0.521 | 0.498 | 0.788 | 0.622 | 0.748 | 0.212 | 0.303 | 0.511 |

## VAL metrics - rule: safety
| model | accuracy | balanced_accuracy | macro_f1 | hazard_recall | hazard_precision | hazard_f2 | miss_rate | false_alarm_rate | severe_miss_irregular_as_firm |
|---|---|---|---|---|---|---|---|---|---|
| LogReg | 0.499 | 0.437 | 0.373 | 0.951 | 0.466 | 0.787 | 0.049 | 0.690 | 0.111 |
| RandomForest | 0.637 | 0.499 | 0.458 | 0.955 | 0.566 | 0.839 | 0.045 | 0.464 | 0.000 |
| CNN | 0.520 | 0.446 | 0.391 | 0.951 | 0.473 | 0.791 | 0.049 | 0.670 | 0.111 |

## TEST metrics - rule: argmax
| model | accuracy | balanced_accuracy | macro_f1 | hazard_recall | hazard_precision | hazard_f2 | miss_rate | false_alarm_rate | severe_miss_irregular_as_firm |
|---|---|---|---|---|---|---|---|---|---|
| Majority | 0.602 | 0.333 | 0.250 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| LogReg | 0.635 | 0.471 | 0.452 | 0.704 | 0.576 | 0.674 | 0.296 | 0.343 | 0.456 |
| RandomForest | 0.692 | 0.504 | 0.478 | 0.734 | 0.642 | 0.714 | 0.266 | 0.270 | 0.474 |
| CNN | 0.669 | 0.482 | 0.459 | 0.668 | 0.618 | 0.657 | 0.332 | 0.273 | 0.544 |

## TEST metrics - rule: safety
| model | accuracy | balanced_accuracy | macro_f1 | hazard_recall | hazard_precision | hazard_f2 | miss_rate | false_alarm_rate | severe_miss_irregular_as_firm |
|---|---|---|---|---|---|---|---|---|---|
| LogReg | 0.525 | 0.445 | 0.386 | 0.957 | 0.495 | 0.806 | 0.043 | 0.646 | 0.123 |
| RandomForest | 0.618 | 0.487 | 0.441 | 0.884 | 0.552 | 0.789 | 0.116 | 0.475 | 0.351 |
| CNN | 0.472 | 0.402 | 0.351 | 0.894 | 0.462 | 0.753 | 0.106 | 0.688 | 0.140 |

## TEST 95 % session-bootstrap CIs (argmax rule)
| model | macro_f1 | macro_f1_lo | macro_f1_hi | hazard_recall | hazard_recall_lo | hazard_recall_hi |
|---|---|---|---|---|---|---|
| Majority | 0.250 | 0.167 | 0.306 | 0.000 | 0.000 | 0.000 |
| LogReg | 0.452 | 0.317 | 0.538 | 0.704 | 0.545 | 0.915 |
| RandomForest | 0.478 | 0.337 | 0.559 | 0.734 | 0.592 | 0.874 |
| CNN | 0.459 | 0.319 | 0.540 | 0.668 | 0.486 | 0.845 |

## TEST 95 % session-bootstrap CIs (safety rule)
| model | macro_f1 | macro_f1_lo | macro_f1_hi | hazard_recall | hazard_recall_lo | hazard_recall_hi |
|---|---|---|---|---|---|---|
| LogReg | 0.386 | 0.233 | 0.476 | 0.957 | 0.902 | 1.000 |
| RandomForest | 0.441 | 0.281 | 0.544 | 0.884 | 0.732 | 0.989 |
| CNN | 0.351 | 0.214 | 0.427 | 0.894 | 0.816 | 0.969 |

## Safety thresholds chosen on VALIDATION (tau, target reached?)
| index | tau | target_reached |
|---|---|---|
| LogReg | 0.860 | True |
| RandomForest | 0.750 | True |
| CNN | 0.920 | True |

## TEST per-tier precision / recall / F1 (argmax)
| model | precision_Firm | recall_Firm | f1_Firm | precision_Compliant | recall_Compliant | f1_Compliant | precision_Irregular | recall_Irregular | f1_Irregular |
|---|---|---|---|---|---|---|---|---|---|
| Majority | 0.602 | 1.000 | 0.751 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| LogReg | 0.771 | 0.657 | 0.709 | 0.542 | 0.738 | 0.625 | 0.028 | 0.018 | 0.022 |
| RandomForest | 0.806 | 0.730 | 0.766 | 0.581 | 0.783 | 0.667 | 0.000 | 0.000 | 0.000 |
| CNN | 0.768 | 0.727 | 0.747 | 0.563 | 0.717 | 0.631 | 0.000 | 0.000 | 0.000 |

## Selected model (best validation macro-F1): **CNN**

### Error analysis - by surface
| surface | tier | n | errors | error_rate |
|---|---|---|---|---|
| hard_tiles_large_space | 2 | 57 | 57 | 1.000 |
| hard_tiles | 0 | 21 | 15 | 0.714 |
| carpet | 1 | 11 | 6 | 0.545 |
| wood | 0 | 201 | 87 | 0.433 |
| soft_pvc | 1 | 176 | 62 | 0.352 |
| tiled | 0 | 36 | 7 | 0.194 |
| fine_concrete | 0 | 70 | 6 | 0.086 |
| concrete | 0 | 127 | 9 | 0.071 |
| soft_tiles | 1 | 57 | 1 | 0.018 |

### Error analysis - hardest sessions
| group_id | surface | tier | n | errors | error_rate |
|---|---|---|---|---|---|
| 24 | wood | 0 | 5 | 5 | 1.000 |
| 33 | hard_tiles_large_space | 2 | 57 | 57 | 1.000 |
| 27 | hard_tiles | 0 | 21 | 15 | 0.714 |
| 43 | wood | 0 | 70 | 46 | 0.657 |
| 52 | carpet | 1 | 11 | 6 | 0.545 |
| 6 | soft_pvc | 1 | 48 | 23 | 0.479 |
| 46 | wood | 0 | 70 | 30 | 0.429 |
| 34 | soft_pvc | 1 | 57 | 22 | 0.386 |

### Error analysis - confusion pairs
| true | pred | count |
|---|---|---|
| Firm | Compliant | 110 |
| Compliant | Firm | 69 |
| Irregular | Firm | 31 |
| Irregular | Compliant | 26 |
| Firm | Irregular | 14 |

### Error analysis - confidence
```json
{
 "mean_confidence_correct": 0.8346356153488159,
 "mean_confidence_wrong": 0.750789225101471,
 "share_of_errors_in_worst_2_groups": 0.248,
 "n_errors": 250,
 "n": 756
}
```

### Error analysis - feature diagnostic
| confusion | feature | median_errors | median_correct_true_tier | median_correct_pred_tier | looks_like |
|---|---|---|---|---|---|
| Firm -> Compliant (n=110) | wY__max | 0.076 | 0.167 | 0.066 | predicted tier |
| Firm -> Compliant (n=110) | wZ__mean | -0.014 | 0.010 | -0.043 | true tier |
| Firm -> Compliant (n=110) | w_mag__max | 0.172 | 0.424 | 0.165 | predicted tier |
| Firm -> Compliant (n=110) | wZ__min | -0.068 | -0.126 | -0.101 | predicted tier |
| Firm -> Compliant (n=110) | wY__mean | 0.006 | -0.001 | 0.017 | true tier |
| Compliant -> Firm (n=69) | wY__max | 0.066 | 0.066 | 0.167 | true tier |
| Compliant -> Firm (n=69) | wZ__mean | 0.026 | -0.043 | 0.010 | predicted tier |
| Compliant -> Firm (n=69) | w_mag__max | 0.297 | 0.165 | 0.424 | predicted tier |
| Compliant -> Firm (n=69) | wZ__min | -0.037 | -0.101 | -0.126 | true tier |
| Compliant -> Firm (n=69) | wY__mean | -0.006 | 0.017 | -0.001 | predicted tier |

## Random-Forest permutation importance (top 12, validation)
| feature | importance_mean | importance_std |
|---|---|---|
| wY__max | 0.006 | 0.001 |
| wZ__mean | 0.004 | 0.001 |
| w_mag__max | 0.004 | 0.000 |
| wZ__min | 0.004 | 0.001 |
| wY__mean | 0.003 | 0.001 |
| gX__std | 0.003 | 0.002 |
| wZ__max | 0.002 | 0.000 |
| wY__mad_diff | 0.002 | 0.001 |
| aZ__mean | 0.002 | 0.000 |
| w_mag__min | 0.001 | 0.001 |
| gX__mean | 0.001 | 0.001 |
| aY__mean | 0.001 | 0.001 |

## EDA numbers
### Eta-squared (terrain vs session)
| signal | terrain_share | session_only_share | eta2_session_total |
|---|---|---|---|
| orientation_X | 0.159 | 0.753 | 0.912 |
| orientation_Y | 0.050 | 0.747 | 0.797 |
| orientation_Z | 0.045 | 0.743 | 0.789 |
| orientation_W | 0.154 | 0.759 | 0.913 |
| angular_velocity_X | 0.002 | 0.039 | 0.041 |
| angular_velocity_Y | 0.021 | 0.283 | 0.303 |
| angular_velocity_Z | 0.021 | 0.286 | 0.307 |
| linear_acceleration_X | 0.016 | 0.114 | 0.130 |
| linear_acceleration_Y | 0.036 | 0.262 | 0.298 |
| linear_acceleration_Z | 0.030 | 0.193 | 0.224 |
| gX (tilt) | 0.006 | 0.481 | 0.486 |
| gY (tilt) | 0.132 | 0.687 | 0.820 |
| gZ (tilt) | 0.134 | 0.673 | 0.808 |
| a_vert__res_std | 0.393 | 0.201 | 0.593 |
| a_horiz__mean | 0.302 | 0.264 | 0.566 |
| w_mag__mean | 0.092 | 0.201 | 0.293 |

### Feature medians by tier
| index | a_vert__res_std | a_vert__std | a_horiz__mean | a_mag__ptp | w_mag__mean | a_vert__band3 |
|---|---|---|---|---|---|---|
| Firm | 1.6297 | 2.2582 | 2.5698 | 11.8353 | 0.1603 | 0.0099 |
| Compliant | 0.6160 | 0.9006 | 1.2865 | 4.6095 | 0.0918 | 0.0086 |
| Irregular | 3.6054 | 4.8515 | 4.2358 | 22.4997 | 0.2741 | 0.0080 |

### Top channel correlations
| channel_1 | channel_2 | rho |
|---|---|---|
| quat_X | quat_W | 0.996 |
| quat_Y | quat_Z | 0.995 |
| acc_Z | a_vert | -0.944 |
| gyro_Y | gyro_Z | -0.616 |
| acc_Y | acc_Z | 0.339 |
| quat_Y | quat_W | -0.201 |
| quat_Z | a_horiz | 0.177 |
| quat_Y | a_horiz | 0.174 |

PCA explained variance (PC1, PC2): [0.4525969026299599, 0.07500160710494125]

## Inference time per window (ms, CPU, includes feature extraction)
| index | ms_per_window |
|---|---|
| LogReg | 0.3116 |
| RandomForest | 0.4341 |
| CNN | 0.0850 |
CNN parameters: 7,299

## Robustness: grouped 5-fold CV over ALL sessions (mean +/- std over folds)
| ('model', '') | ('macro_f1', 'mean') | ('macro_f1', 'std') | ('hazard_recall', 'mean') | ('hazard_recall', 'std') | ('hazard_precision', 'mean') | ('hazard_precision', 'std') |
|---|---|---|---|---|---|---|
| CNN | 0.570 | 0.121 | 0.759 | 0.060 | 0.621 | 0.050 |
| LogReg | 0.559 | 0.111 | 0.778 | 0.061 | 0.569 | 0.052 |
| RandomForest | 0.629 | 0.156 | 0.810 | 0.100 | 0.659 | 0.076 |

### Out-of-fold share of windows assigned to the correct tier, by surface
| index | LogReg | RandomForest | CNN | n_windows | n_sessions |
|---|---|---|---|---|---|
| carpet | 0.635 | 0.683 | 0.508 | 189 | 4 |
| concrete | 0.657 | 0.805 | 0.747 | 779 | 15 |
| fine_concrete | 0.567 | 0.526 | 0.614 | 363 | 7 |
| hard_tiles | 0.238 | 0.286 | 0.286 | 21 | 1 |
| hard_tiles_large_space | 0.503 | 0.403 | 0.377 | 308 | 5 |
| soft_pvc | 0.717 | 0.863 | 0.747 | 732 | 14 |
| soft_tiles | 0.926 | 0.929 | 0.909 | 297 | 6 |
| tiled | 0.665 | 0.856 | 0.796 | 514 | 9 |
| wood | 0.524 | 0.591 | 0.570 | 607 | 12 |

## Leakage demonstration (5-fold CV, Random Forest, all windows)
| features | protocol | macro_f1 | hazard_recall | accuracy |
|---|---|---|---|---|
| engineered features (used in this project) | naive random folds (LEAKY) | 0.746 | 0.799 | 0.821 |
| engineered features (used in this project) | session-grouped folds (correct) | 0.652 | 0.697 | 0.754 |
| engineered + raw orientation (heading leaks the session) | naive random folds (LEAKY) | 0.874 | 0.868 | 0.895 |
| engineered + raw orientation (heading leaks the session) | session-grouped folds (correct) | 0.657 | 0.655 | 0.766 |
