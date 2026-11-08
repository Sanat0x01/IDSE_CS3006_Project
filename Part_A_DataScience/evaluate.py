"""
evaluate.py -- risk-aware metrics, uncertainty, safety threshold, error analysis, plots.

Why these metrics? (full discussion in the report)
--------------------------------------------------
Positive class = "hazardous terrain" (Compliant or Irregular tier).
* FALSE NEGATIVE (hazard called Firm): the rover keeps full speed/torque on carpet
  or tile gaps -> stall, wheel slip, jolt, possibly a lost mission. EXPENSIVE.
* FALSE POSITIVE (Firm called hazard): the rover slows down needlessly -> lost
  time/energy. CHEAP, but not free (a rover that always brakes is useless).
Therefore: hazard RECALL is the safety metric, hazard PRECISION is the efficiency
metric, F2 weights recall higher, and MACRO-F1 (not accuracy) is the headline score
because the tiers are imbalanced and the rare Irregular tier matters most.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

import config as C
from models import hazard_recall_score

TIER_COLORS = ["#2a9d8f", "#e9c46a", "#e76f51"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(C.TIER_NAMES)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    ht, hp = y_true >= 1, y_pred >= 1
    tp, fp, fn = int((ht & hp).sum()), int((~ht & hp).sum()), int((ht & ~hp).sum())
    h_prec = tp / max(tp + fp, 1)
    h_rec = tp / max(tp + fn, 1)
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f.mean()),
        "hazard_recall": h_rec,
        "hazard_precision": h_prec,
        "hazard_f2": float(5 * h_prec * h_rec / max(4 * h_prec + h_rec, 1e-12)),
        "miss_rate": 1 - h_rec,                                          # hazard predicted Firm
        "false_alarm_rate": float(fp / max((~ht).sum(), 1)),             # Firm predicted hazard
        "severe_miss_irregular_as_firm": float(((y_true == 2) & (y_pred == 0)).sum() / max((y_true == 2).sum(), 1)),
    }
    for i, name in enumerate(C.TIER_NAMES):
        out[f"precision_{name}"], out[f"recall_{name}"], out[f"f1_{name}"] = float(p[i]), float(r[i]), float(f[i])
    return out


def group_bootstrap_ci(y_true, y_pred, groups, metric_fn, n_boot: int = 1000, seed: int = C.SEED):
    """95 % CI by resampling whole SESSIONS (series in a session are not independent)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def macro_f1_fn(yt, yp) -> float:
    return f1_score(yt, yp, average="macro", labels=list(range(len(C.TIER_NAMES))), zero_division=0)


# --------------------------------------------------------------------------- #
# Safety-biased decision rule
# --------------------------------------------------------------------------- #
def safety_predict(proba: np.ndarray, tau: float) -> np.ndarray:
    """Predict Firm only if P(Firm) >= tau; otherwise the more likely hazard tier."""
    haz = proba[:, 1:].argmax(1) + 1
    return np.where(proba[:, 0] >= tau, 0, haz)


def choose_safety_threshold(proba_val, y_val, target: float = C.SAFETY_RECALL_TARGET) -> tuple[float, bool]:
    """Smallest tau >= 0.5 whose hazard recall on the VALIDATION sessions reaches the target."""
    for tau in np.round(np.arange(0.50, 0.981, 0.01), 2):
        if hazard_recall_score(y_val, safety_predict(proba_val, tau)) >= target:
            return float(tau), True
    return 0.98, False


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
def error_analysis(meta: pd.DataFrame, feats: pd.DataFrame, y_true, y_pred, proba, top_features: list[str]):
    df = meta.reset_index(drop=True).copy()
    df["true"], df["pred"] = y_true, y_pred
    df["confidence"] = proba.max(1)
    df["correct"] = df["true"] == df["pred"]

    by_surface = (df.groupby(["surface", "tier"]).agg(n=("correct", "size"), errors=("correct", lambda s: int((~s).sum())))
                  .reset_index())
    by_surface["error_rate"] = by_surface["errors"] / by_surface["n"]
    by_surface = by_surface.sort_values("error_rate", ascending=False)

    by_group = (df.groupby(["group_id", "surface", "tier"]).agg(n=("correct", "size"), errors=("correct", lambda s: int((~s).sum())))
                .reset_index())
    by_group["error_rate"] = by_group["errors"] / by_group["n"]
    by_group = by_group.sort_values("error_rate", ascending=False)

    pairs = (df[~df["correct"]].groupby(["true", "pred"]).size().rename("count").reset_index()
             .sort_values("count", ascending=False))
    pairs["true"] = pairs["true"].map(dict(enumerate(C.TIER_NAMES)))
    pairs["pred"] = pairs["pred"].map(dict(enumerate(C.TIER_NAMES)))

    conf = {"mean_confidence_correct": float(df.loc[df.correct, "confidence"].mean()),
            "mean_confidence_wrong": float(df.loc[~df.correct, "confidence"].mean()) if (~df.correct).any() else float("nan"),
            "share_of_errors_in_worst_2_groups": float(by_group["errors"].head(2).sum() / max(by_group["errors"].sum(), 1)),
            "n_errors": int((~df.correct).sum()), "n": int(len(df))}

    # Feature diagnostic: are the misclassified windows physically closer to the tier they were
    # (wrongly) assigned to than to the tier they truly belong to?
    rows = []
    F = feats.reset_index(drop=True)
    for (t, pr), cnt in df[~df.correct].groupby(["true", "pred"]).size().sort_values(ascending=False).head(3).items():
        mask_err = (df.true == t) & (df.pred == pr)
        for feat in top_features:
            med_err = float(F.loc[mask_err, feat].median())
            ref_true = F.loc[(df.true == t) & df.correct, feat]
            ref_pred = F.loc[(df.true == pr) & df.correct, feat]
            if len(ref_true) == 0 or len(ref_pred) == 0:
                continue
            m_t, m_p = float(ref_true.median()), float(ref_pred.median())
            rows.append({"confusion": f"{C.TIER_NAMES[t]} -> {C.TIER_NAMES[pr]} (n={cnt})", "feature": feat,
                         "median_errors": med_err, "median_correct_true_tier": m_t, "median_correct_pred_tier": m_p,
                         "looks_like": "predicted tier" if abs(med_err - m_p) < abs(med_err - m_t) else "true tier"})
    return {"windows": df, "by_surface": by_surface, "by_group": by_group, "pairs": pairs,
            "confidence": conf, "feature_diagnostic": pd.DataFrame(rows)}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrices(preds: dict, y_true, path):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.6 * len(preds), 4.2))
    axes = np.atleast_1d(axes)
    labels = list(range(len(C.TIER_NAMES)))
    for ax, (name, yp) in zip(axes, preds.items()):
        cm = confusion_matrix(y_true, yp, labels=labels)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        annot = np.array([[f"{norm[i, j]:.0%}\n(n={cm[i, j]})" for j in labels] for i in labels])
        sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=C.TIER_NAMES, yticklabels=C.TIER_NAMES)
        ax.set_title(name)
        ax.set_xlabel("Predicted tier")
        ax.set_ylabel("True tier")
    fig.suptitle("Test-set confusion matrices (row-normalised; unseen sessions)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(table: pd.DataFrame, path):
    """table: index = model, columns macro_f1, macro_f1_lo, macro_f1_hi, hazard_recall, hazard_precision."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    metrics = ["macro_f1", "hazard_recall", "hazard_precision"]
    width = 0.8 / len(metrics)
    x = np.arange(len(table))
    for i, m in enumerate(metrics):
        err = None
        if m == "macro_f1" and {"macro_f1_lo", "macro_f1_hi"} <= set(table.columns):
            err = np.vstack([table[m] - table["macro_f1_lo"], table["macro_f1_hi"] - table[m]])
        ax.bar(x + (i - 1) * width, table[m], width, label=m.replace("_", " "), yerr=err, capsize=3)
    ax.axhline(C.GOALS["hazard_recall"], ls="--", c="k", lw=0.8)
    ax.text(len(table) - 0.45, C.GOALS["hazard_recall"] + 0.01, "recall goal", ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=0)
    ax.set_ylim(0, 1.08)
    ax.set_title("Held-out TEST sessions: model comparison (error bars = 95 % session-bootstrap CI)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cnn_history(history: list[dict], path):
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(h["epoch"], h["train_loss"], label="train loss (weighted CE)")
    if "val_loss" in h:
        ax[0].plot(h["epoch"], h["val_loss"], label="validation loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[0].set_title("CNN learning curves")
    if "val_macro_f1" in h:
        ax[1].plot(h["epoch"], h["val_macro_f1"], c="C2")
        best = h["val_macro_f1"].idxmax()
        ax[1].scatter([h.loc[best, "epoch"]], [h.loc[best, "val_macro_f1"]], c="r", zorder=3, label="selected checkpoint")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("validation macro-F1")
        ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_analysis(ea: dict, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bs = ea["by_surface"].sort_values("error_rate")
    ax[0].barh(bs["surface"], bs["error_rate"], color=[TIER_COLORS[t] for t in bs["tier"]])
    for i, (e, n) in enumerate(zip(bs["errors"], bs["n"])):
        ax[0].text(0.005, i, f"{e}/{n}", va="center", fontsize=8)
    ax[0].set_xlabel("misclassification rate")
    ax[0].set_title("Errors by floor surface (colour = true tier)")
    bg = ea["by_group"].head(10).iloc[::-1]
    ax[1].barh([f"session {g} ({s})" for g, s in zip(bg["group_id"], bg["surface"])], bg["error_rate"],
               color=[TIER_COLORS[t] for t in bg["tier"]])
    ax[1].set_xlabel("misclassification rate")
    ax[1].set_title("10 hardest test sessions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_leakage_demo(leak: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    feats = list(dict.fromkeys(leak["features"]))
    protos = list(dict.fromkeys(leak["protocol"]))
    w = 0.38
    for i, p in enumerate(protos):
        vals = [leak[(leak.features == f) & (leak.protocol == p)]["macro_f1"].iloc[0] for f in feats]
        bars = ax.bar(np.arange(len(feats)) + (i - 0.5) * w, vals, w, label=p, color=["#bc4749", "#2a9d8f"][i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(feats)))
    ax.set_xticklabels([f.replace(" (", "\n(") for f in feats], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("macro-F1 (5-fold CV)")
    ax.set_title("Leakage demonstration: random folds inflate the score when features reveal the session")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, path, title: str):
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#457b9d")
    ax.set_xlabel("drop in validation macro-F1 when the feature is shuffled")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Small helper: DataFrame -> Markdown table (no extra dependency needed)
# --------------------------------------------------------------------------- #
def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.3f}", index: bool = True) -> str:
    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
