"""
run_pipeline.py -- end-to-end terrain-risk classification pipeline.

    python run_pipeline.py              # full run (about 10-15 min on a laptop CPU)
    python run_pipeline.py --quick      # smoke test (~1-2 min, smaller grids; numbers NOT for the report)
    python run_pipeline.py --skip-robust

Order of operations (this order is what guarantees zero leakage):
  1. load + audit/clean the data           (label-free, row-wise)
  2. split SESSIONS into train / val / test (test is not touched until step 8)
  3. EDA on TRAIN only
  4. grouped CV on TRAIN -> hyper-parameter tuning of LogReg / Random Forest / CNN
  5. final models fitted on TRAIN (CNN checkpoint chosen on VAL)
  6. VAL: model selection + safety-threshold choice
  7. TEST: one-shot evaluation, bootstrap CIs, error analysis
  8. robustness check: grouped 5-fold CV over ALL sessions (every session tested once)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

import config as C
import data as D
import evaluate as E
import features as F
import models as M
from eda import run_eda


def _save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _time_it(fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def leakage_demo(feats: pd.DataFrame, cube: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """DELIBERATELY WRONG protocol shown next to the right one (demonstration only, never used
    for model selection): 5-fold CV over all windows with (a) naive random folds and
    (b) session-grouped folds, with and without the raw orientation quaternion statistics."""
    y, g = meta["tier"].to_numpy(), meta["group_id"].to_numpy()
    ori = np.hstack([cube[:, :, :4].mean(1), cube[:, :, :4].std(1)])             # 8 heading-dependent features
    feature_sets = {"engineered features (used in this project)": feats.to_numpy(),
                    "engineered + raw orientation (heading leaks the session)": np.hstack([feats.to_numpy(), ori])}
    protocols = {"naive random folds (LEAKY)": list(StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(np.zeros(len(y)), y)),
                 "session-grouped folds (correct)": D.grouped_cv_folds(y, g, 5)}
    rf = M.build_rf().set_params(n_estimators=200, min_samples_leaf=3, max_features="sqrt")
    rows = []
    for fname, Xf in feature_sets.items():
        for pname, folds in protocols.items():
            oof = np.zeros(len(y), dtype=int)
            for a, b in folds:
                oof[b] = clone(rf).fit(Xf[a], y[a]).predict(Xf[b])
            m = E.compute_metrics(y, oof)
            rows.append({"features": fname, "protocol": pname, "macro_f1": m["macro_f1"],
                         "hazard_recall": m["hazard_recall"], "accuracy": m["accuracy"]})
    return pd.DataFrame(rows)


def main(args) -> None:
    t_start = time.time()
    M.set_global_seed(C.SEED)                                   # <- all seeds fixed here
    for d in (C.FIG_DIR, C.RES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    quick = args.quick
    print(f"=== Terrain-risk pipeline (seed={C.SEED}{', QUICK MODE' if quick else ''}) ===")

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    X_raw, y_raw = D.load_raw(Path(args.data_dir))
    cube, meta, audit = D.validate_and_clean(X_raw, y_raw)
    del X_raw, y_raw
    _save_json(audit, C.RES_DIR / "data_audit.json")
    print(f"[1] data: {audit['series_out']} windows x {C.N_STEPS} steps x {len(C.RAW_CHANNELS)} channels, "
          f"{audit['groups']} sessions")

    feats = F.extract_features(cube)                           # per-window, label-free -> no leakage
    seq = F.build_sequences(cube)
    y_all, g_all = meta["tier"].to_numpy(), meta["group_id"].to_numpy()

    # ------------------------------------------------------------------ #
    # 2. Split by session
    # ------------------------------------------------------------------ #
    sp = D.make_group_splits(meta)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    sp["manifest"].to_csv(C.RES_DIR / "split_manifest.csv", index=False)
    split_summary = pd.DataFrame({
        s: {"windows": len(i), "sessions": int(meta.iloc[i].group_id.nunique()),
            **{f"{n} windows": int((y_all[i] == k).sum()) for k, n in enumerate(C.TIER_NAMES)},
            **{f"{n} sessions": int(meta.iloc[i][meta.iloc[i].tier == k].group_id.nunique()) for k, n in enumerate(C.TIER_NAMES)}}
        for s, i in dict(train=tr, val=va, test=te).items()}).T
    split_summary.to_csv(C.RES_DIR / "split_summary.csv")
    print("[2] session-level split (zero overlap asserted):\n", split_summary.to_string())

    # ------------------------------------------------------------------ #
    # 3. EDA (train only)
    # ------------------------------------------------------------------ #
    eda = run_eda(cube[tr], meta.iloc[tr], feats.iloc[tr], C.FIG_DIR)
    for k in ("eta_squared", "feature_medians", "feature_iqr", "top_correlations", "surface_counts"):
        eda[k].to_csv(C.RES_DIR / f"eda_{k}.csv")
    print("[3] EDA figures written to", C.FIG_DIR)

    # ------------------------------------------------------------------ #
    # 4. Tuning with grouped CV on TRAIN
    # ------------------------------------------------------------------ #
    folds = D.grouped_cv_folds(y_all[tr], g_all[tr], C.CV_FOLDS)
    Xtr, Xva, Xte = (feats.iloc[i].to_numpy() for i in (tr, va, te))
    ytr, yva, yte = y_all[tr], y_all[va], y_all[te]
    fitted, cv_tables, best_params = {}, {}, {}

    fitted["Majority"] = M.build_dummy().fit(Xtr, ytr)
    print("[4] tuning (grouped CV on TRAIN sessions)")
    lr, best_params["LogReg"], cv_tables["LogReg"] = M.tune_sklearn(M.build_logreg(), M.LOGREG_GRID, Xtr, ytr, folds, "grid")
    print("   LogReg best:", best_params["LogReg"])
    fitted["LogReg"] = lr
    rf, best_params["RandomForest"], cv_tables["RandomForest"] = M.tune_sklearn(
        M.build_rf(), M.RF_SPACE, Xtr, ytr, folds, "random", n_iter=4 if quick else 12)
    print("   RandomForest best:", best_params["RandomForest"])
    fitted["RandomForest"] = rf

    cnn_grid = M.CNN_GRID[:2] if quick else M.CNN_GRID
    cnn_epochs = 8 if quick else 30
    best_cfg, cv_tables["CNN"] = M.tune_cnn(seq[tr], ytr, folds, cnn_grid, cnn_epochs)
    best_params["CNN"] = best_cfg
    print("   CNN best:", best_cfg)
    for name, tab in cv_tables.items():
        tab.to_csv(C.RES_DIR / f"cv_results_{name}.csv", index=False)
    _save_json(best_params, C.RES_DIR / "best_params.json")

    # ------------------------------------------------------------------ #
    # 5. Final models on TRAIN (CNN: checkpoint chosen on VAL)
    # ------------------------------------------------------------------ #
    scaler = F.SequenceScaler().fit(seq[tr])
    s_tr, s_va, s_te = (scaler.transform(seq[i]) for i in (tr, va, te))
    cnn = M.TorchTerrainCNN(n_in=seq.shape[1], epochs=cnn_epochs, **best_cfg).fit(s_tr, ytr, s_va, yva)
    fitted["CNN"] = cnn
    E.plot_cnn_history(cnn.history_, C.FIG_DIR / "fig07_cnn_learning_curves.png")
    pd.DataFrame(cnn.history_).to_csv(C.RES_DIR / "cnn_history.csv", index=False)

    def proba(name, split):
        if name == "CNN":
            return cnn.predict_proba({"val": s_va, "test": s_te}[split])
        return fitted[name].predict_proba({"val": Xva, "test": Xte}[split])

    names = list(fitted)
    P = {(n, s): proba(n, s) for n in names for s in ("val", "test")}
    truth = {"val": yva, "test": yte}
    groups = {"val": g_all[va], "test": g_all[te]}

    # ------------------------------------------------------------------ #
    # 6/7. Evaluation (VAL -> selection + threshold, TEST -> one-shot report)
    # ------------------------------------------------------------------ #
    taus = {n: E.choose_safety_threshold(P[(n, "val")], yva) for n in names if n != "Majority"}
    rows = []
    for n in names:
        for s in ("val", "test"):
            variants = {"argmax": P[(n, s)].argmax(1)}
            if n != "Majority":
                variants[f"safety(tau={taus[n][0]:.2f})"] = E.safety_predict(P[(n, s)], taus[n][0])
            for rule, yp in variants.items():
                m = E.compute_metrics(truth[s], yp)
                m.update(model=n, split=s, rule=rule.split("(")[0])
                if s == "test":
                    for key, fn in (("macro_f1", E.macro_f1_fn), ("hazard_recall", M.hazard_recall_score)):
                        lo, hi = E.group_bootstrap_ci(truth[s], yp, groups[s], fn, n_boot=200 if quick else 1000)
                        m[f"{key}_lo"], m[f"{key}_hi"] = lo, hi
                rows.append(m)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(C.RES_DIR / "metrics_all.csv", index=False)

    val_argmax = metrics[(metrics.split == "val") & (metrics.rule == "argmax") & (metrics.model != "Majority")]
    selected = val_argmax.sort_values("macro_f1", ascending=False).iloc[0]["model"]   # chosen on VAL, not TEST
    print(f"[6] model selected on VALIDATION macro-F1: {selected}")

    test_arg = metrics[(metrics.split == "test") & (metrics.rule == "argmax")].set_index("model")
    test_saf = metrics[(metrics.split == "test") & (metrics.rule == "safety")].set_index("model")
    E.plot_model_comparison(test_arg.loc[[n for n in names if n != "Majority"]],
                            C.FIG_DIR / "fig08_model_comparison.png")
    E.plot_confusion_matrices({n: P[(n, "test")].argmax(1) for n in names if n != "Majority"}, yte,
                              C.FIG_DIR / "fig09_confusion_matrices.png")

    # Interpretability: permutation importance of the Random Forest on VAL sessions
    pi = permutation_importance(fitted["RandomForest"], Xva, yva, scoring="f1_macro",
                                n_repeats=1 if quick else 3, random_state=C.SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": feats.columns, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
    imp.to_csv(C.RES_DIR / "rf_permutation_importance.csv", index=False)
    E.plot_importance(imp, C.FIG_DIR / "fig10_rf_permutation_importance.png",
                      "Random Forest: permutation importance (validation sessions)")

    # Error analysis of the model that was selected on validation
    sel_pred = P[(selected, "test")].argmax(1)
    ea = E.error_analysis(meta.iloc[te], feats.iloc[te], yte, sel_pred, P[(selected, "test")],
                          top_features=list(imp["feature"].head(5)))
    E.plot_error_analysis(ea, C.FIG_DIR / "fig11_error_analysis.png")
    ea["windows"].to_csv(C.RES_DIR / "test_predictions_selected_model.csv", index=False)
    for k in ("by_surface", "by_group", "pairs", "feature_diagnostic"):
        ea[k].to_csv(C.RES_DIR / f"error_{k}.csv", index=False)

    # Inference cost (feature extraction included for the feature-based models)
    cube_te = cube[te]
    timing = {
        "LogReg": _time_it(lambda: fitted["LogReg"].predict(F.extract_features(cube_te).to_numpy())),
        "RandomForest": _time_it(lambda: fitted["RandomForest"].predict(F.extract_features(cube_te).to_numpy())),
        "CNN": _time_it(lambda: cnn.predict(scaler.transform(F.build_sequences(cube_te)))),
    }
    timing = {k: 1000 * v / len(te) for k, v in timing.items()}

    # ------------------------------------------------------------------ #
    # 8. Robustness: grouped 5-fold CV over ALL sessions (each session tested once)
    # ------------------------------------------------------------------ #
    robust_rows, oof = [], {}
    if not args.skip_robust and not quick:
        print("[8] robustness check: grouped 5-fold CV over all sessions")
        outer = D.grouped_cv_folds(y_all, g_all, C.ROBUST_CV_FOLDS, seed=C.SEED + 1000)
        oof = {n: np.full(len(y_all), -1) for n in ("LogReg", "RandomForest", "CNN")}
        for k, (a, b) in enumerate(outer):
            for n in oof:
                if n == "CNN":
                    sc = F.SequenceScaler().fit(seq[a])
                    net = M.TorchTerrainCNN(n_in=seq.shape[1], epochs=cnn_epochs, **best_cfg).fit(sc.transform(seq[a]), y_all[a])
                    pred = net.predict(sc.transform(seq[b]))
                else:
                    pred = clone(fitted[n]).fit(feats.iloc[a].to_numpy(), y_all[a]).predict(feats.iloc[b].to_numpy())
                oof[n][b] = pred
                m = E.compute_metrics(y_all[b], pred)
                robust_rows.append({"model": n, "fold": k, **{q: m[q] for q in ("macro_f1", "hazard_recall", "hazard_precision", "balanced_accuracy")}})
        robust = pd.DataFrame(robust_rows)
        robust.to_csv(C.RES_DIR / "robust_cv_folds.csv", index=False)
        pooled = pd.DataFrame({n: E.compute_metrics(y_all, p) for n, p in oof.items()}).T
        pooled.to_csv(C.RES_DIR / "robust_cv_pooled.csv")
        per_surface = pd.DataFrame({n: pd.Series((oof[n] == y_all)).groupby(meta["surface"].to_numpy()).mean() for n in oof})
        per_surface["n_windows"] = meta.groupby("surface").size()
        per_surface["n_sessions"] = meta.groupby("surface")["group_id"].nunique()
        per_surface.to_csv(C.RES_DIR / "robust_cv_per_surface.csv")

    # ------------------------------------------------------------------ #
    # 9. Leakage demonstration (why the split is by session)
    # ------------------------------------------------------------------ #
    leak = leakage_demo(feats, cube, meta)
    leak.to_csv(C.RES_DIR / "leakage_demo.csv", index=False)
    E.plot_leakage_demo(leak, C.FIG_DIR / "fig12_leakage_demo.png")
    print("[9] leakage demonstration:\n", leak.round(3).to_string(index=False))

    # ------------------------------------------------------------------ #
    # Machine-written summary (numbers that the report quotes)
    # ------------------------------------------------------------------ #
    write_summary(audit, split_summary, eda, best_params, cv_tables, metrics, taus, selected, test_arg, test_saf,
                  imp, ea, timing, cnn, robust_rows, oof, meta, y_all, quick, leak)

    sel_t = test_arg.loc[selected]
    sel_s = test_saf.loc[selected]
    print("\n=== GOAL CHECK (selected model on held-out TEST sessions) ===")
    for tag, row in (("argmax rule", sel_t), ("safety rule ", sel_s)):
        print(f"  {selected} [{tag}] hazard recall {row.hazard_recall:.3f} "
              f"(goal >= {C.GOALS['hazard_recall']}: {'MET' if row.hazard_recall >= C.GOALS['hazard_recall'] else 'NOT met'}) | "
              f"macro-F1 {row.macro_f1:.3f} (goal >= {C.GOALS['macro_f1']}: {'MET' if row.macro_f1 >= C.GOALS['macro_f1'] else 'NOT met'})")
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. See {C.OUT_DIR}")


# --------------------------------------------------------------------------- #
def write_summary(audit, split_summary, eda, best_params, cv_tables, metrics, taus, selected, test_arg, test_saf,
                  imp, ea, timing, cnn, robust_rows, oof, meta, y_all, quick, leak) -> None:
    md = E.df_to_md
    L = ["# Auto-generated results summary", "",
         "_Written by `run_pipeline.py`; every number in REPORT.md comes from a run like this one._",
         "_QUICK MODE: do not quote these numbers._" if quick else "", ""]
    L += ["## Data audit", "```json", json.dumps({k: v for k, v in audit.items() if k not in ("channel_min", "channel_max")}, indent=1), "```", ""]
    L += ["## Split (by session)", md(split_summary, "{:.0f}"), ""]
    L += ["## Best hyper-parameters (grouped CV on TRAIN)", "```json", json.dumps(best_params, indent=1, default=str), "```", ""]
    for n, t in cv_tables.items():
        L += [f"### CV results - {n} (top 5)", md(t.head(5).assign(params=lambda d: d.params.astype(str)), index=False), ""]
    cols = ["accuracy", "balanced_accuracy", "macro_f1", "hazard_recall", "hazard_precision", "hazard_f2",
            "miss_rate", "false_alarm_rate", "severe_miss_irregular_as_firm"]
    for s in ("val", "test"):
        for rule in ("argmax", "safety"):
            sub = metrics[(metrics.split == s) & (metrics.rule == rule)].set_index("model")[cols]
            if len(sub):
                L += [f"## {s.upper()} metrics - rule: {rule}", md(sub), ""]
    ci_cols = ["macro_f1", "macro_f1_lo", "macro_f1_hi", "hazard_recall", "hazard_recall_lo", "hazard_recall_hi"]
    L += ["## TEST 95 % session-bootstrap CIs (argmax rule)", md(test_arg[ci_cols]), "",
          "## TEST 95 % session-bootstrap CIs (safety rule)", md(test_saf[ci_cols]), ""]
    L += ["## Safety thresholds chosen on VALIDATION (tau, target reached?)",
          md(pd.DataFrame({k: {"tau": v[0], "target_reached": v[1]} for k, v in taus.items()}).T), ""]
    pc = [f"{c}" for c in ("precision_Firm", "recall_Firm", "f1_Firm", "precision_Compliant", "recall_Compliant", "f1_Compliant",
                           "precision_Irregular", "recall_Irregular", "f1_Irregular")]
    L += ["## TEST per-tier precision / recall / F1 (argmax)", md(test_arg[pc]), ""]
    L += [f"## Selected model (best validation macro-F1): **{selected}**", "",
          "### Error analysis - by surface", md(ea["by_surface"], index=False), "",
          "### Error analysis - hardest sessions", md(ea["by_group"].head(8), index=False), "",
          "### Error analysis - confusion pairs", md(ea["pairs"], "{:.0f}", index=False), "",
          "### Error analysis - confidence", "```json", json.dumps(ea["confidence"], indent=1), "```", ""]
    if len(ea["feature_diagnostic"]):
        L += ["### Error analysis - feature diagnostic", md(ea["feature_diagnostic"], "{:.3f}", index=False), ""]
    L += ["## Random-Forest permutation importance (top 12, validation)", md(imp.head(12), index=False), ""]
    L += ["## EDA numbers", "### Eta-squared (terrain vs session)", md(eda["eta_squared"]), "",
          "### Feature medians by tier", md(eda["feature_medians"], "{:.4f}"), "",
          "### Top channel correlations", md(eda["top_correlations"], index=False), "",
          f"PCA explained variance (PC1, PC2): {eda['pca_var_ratio']}", ""]
    L += ["## Inference time per window (ms, CPU, includes feature extraction)", md(pd.Series(timing, name="ms_per_window").to_frame(), "{:.4f}"),
          f"CNN parameters: {cnn.n_parameters():,}", ""]
    if robust_rows:
        r = pd.DataFrame(robust_rows)
        L += ["## Robustness: grouped 5-fold CV over ALL sessions (mean +/- std over folds)",
              md(r.groupby("model")[["macro_f1", "hazard_recall", "hazard_precision"]].agg(["mean", "std"])), ""]
        ps = pd.read_csv(C.RES_DIR / "robust_cv_per_surface.csv", index_col=0)
        L += ["### Out-of-fold share of windows assigned to the correct tier, by surface", md(ps), ""]
    L += ["## Leakage demonstration (5-fold CV, Random Forest, all windows)", md(leak, index=False), ""]
    (C.RES_DIR / "results_summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test with tiny grids (numbers not for the report)")
    ap.add_argument("--skip-robust", action="store_true", help="skip the 5-fold grouped CV over all sessions")
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    main(ap.parse_args())
