"""
eda.py -- exploratory data analysis on the TRAIN sessions only.

Doing EDA on the training split (not the full data) is deliberate: looking at
test-session statistics while designing features would be a subtle form of leakage.
Every function returns plain numbers/tables so the report can quote them.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config as C
import features as F
from evaluate import TIER_COLORS

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = dict(zip(C.TIER_NAMES, TIER_COLORS))


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """Share of total variance explained by a categorical grouping (ANOVA eta^2)."""
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    ss_between = sum(((values[labels == g].mean() - grand) ** 2) * (labels == g).sum() for g in np.unique(labels))
    return float(ss_between / max(ss_tot, 1e-12))


def run_eda(cube: np.ndarray, meta: pd.DataFrame, feats: pd.DataFrame, fig_dir, seed: int = C.SEED) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tier_name = meta["tier"].map(dict(enumerate(C.TIER_NAMES))).to_numpy()
    out: dict = {}
    sig = F.derive_signals(cube)

    # ------------------------------------------------------------------ #
    # Plot 1 - class balance: series per tier / per surface (+ sessions)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    tc = meta["tier"].value_counts().sort_index()
    ax[0].bar(C.TIER_NAMES, tc.values, color=TIER_COLORS)
    for i, v in enumerate(tc.values):
        ax[0].text(i, v, f"{v}\n({v / tc.sum():.0%})", ha="center", va="bottom")
    ax[0].set_ylabel("series (128-sample windows)")
    ax[0].set_title("Navigation-risk tiers (TRAIN)")
    ax[0].set_ylim(0, tc.max() * 1.2)
    sc = meta.groupby(["surface", "tier"]).agg(n=("series_id", "size"), sessions=("group_id", "nunique")).reset_index()
    sc = sc.sort_values("n")
    ax[1].barh(sc["surface"], sc["n"], color=[TIER_COLORS[t] for t in sc["tier"]])
    for i, (n, s) in enumerate(zip(sc["n"], sc["sessions"])):
        ax[1].text(n, i, f" {n} ({s} sess.)", va="center", fontsize=8)
    ax[1].set_xlim(0, sc["n"].max() * 1.3)
    ax[1].set_title("Floor surfaces (colour = tier)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_class_balance.png", dpi=150)
    plt.close(fig)
    out["tier_counts"] = {C.TIER_NAMES[k]: int(v) for k, v in tc.items()}
    out["surface_counts"] = sc

    # ------------------------------------------------------------------ #
    # Plot 2 - correlation heatmap of raw + derived channels
    # ------------------------------------------------------------------ #
    flat = cube.reshape(-1, cube.shape[2])
    pick = rng.choice(len(flat), size=min(60000, len(flat)), replace=False)
    frame = pd.DataFrame(flat[pick], columns=[c.replace("linear_acceleration", "acc").replace("angular_velocity", "gyro")
                                              .replace("orientation", "quat") for c in C.RAW_CHANNELS])
    frame["a_vert"] = sig["a_vert"].reshape(-1)[pick]
    frame["a_horiz"] = sig["a_horiz"].reshape(-1)[pick]
    corr = frame.corr(method="spearman")      # Spearman: robust to the heavy-tailed spikes
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7})
    ax.set_title("Spearman correlation of IMU channels (60k random TRAIN samples)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    tri = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().rename("rho").reset_index()
    tri.columns = ["channel_1", "channel_2", "rho"]
    out["top_correlations"] = tri.reindex(tri["rho"].abs().sort_values(ascending=False).index).head(8)

    # ------------------------------------------------------------------ #
    # Plot 3 - example signals per tier (vertical vibration + rotation rate)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(len(C.TIER_NAMES), 2, figsize=(12, 7), sharex=True)
    for t, name in enumerate(C.TIER_NAMES):
        idx_t = np.flatnonzero(meta["tier"].to_numpy() == t)
        groups_t = meta["group_id"].to_numpy()[idx_t]
        chosen = []                                           # three series from three different sessions
        for g in rng.permutation(np.unique(groups_t))[:3]:
            chosen.append(rng.choice(idx_t[groups_t == g]))
        for k, i in enumerate(chosen):
            lab = meta["surface"].iloc[i]
            axes[t, 0].plot(sig["a_vert"][i] - sig["a_vert"][i].mean(), lw=0.9, label=lab)
            axes[t, 1].plot(sig["w_mag"][i], lw=0.9, label=lab)
        axes[t, 0].set_ylabel(f"{name}\nvertical acc. (m/s^2)", color=TIER_COLORS[t])
        axes[t, 1].set_ylabel("|angular velocity| (rad/s)")
        axes[t, 0].legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("sample index within window")
    axes[-1, 1].set_xlabel("sample index within window")
    fig.suptitle("Example windows: vertical acceleration (mean removed) and rotation rate", y=0.995)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_signal_examples.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Plot 4 - engineered-feature distributions by tier
    # ------------------------------------------------------------------ #
    key = ["a_vert__res_std", "a_vert__std", "a_horiz__mean", "a_mag__ptp", "w_mag__mean", "a_vert__band3"]
    key = [k for k in key if k in feats.columns]
    long = feats[key].copy()
    long["tier"] = tier_name
    long = long.melt(id_vars="tier", var_name="feature", value_name="value")
    g = sns.catplot(data=long, x="tier", y="value", col="feature", col_wrap=3, kind="box", order=C.TIER_NAMES,
                    palette=PALETTE, hue="tier", legend=False, sharey=False, showfliers=False, height=3.0, aspect=1.15)
    g.set_titles("{col_name}")
    g.figure.suptitle("Key engineered features by risk tier (TRAIN, outliers hidden)", y=1.02)
    g.savefig(fig_dir / "fig04_feature_distributions.png", dpi=150)
    plt.close(g.figure)
    out["feature_medians"] = feats[key].groupby(tier_name).median().reindex(C.TIER_NAMES)
    out["feature_iqr"] = (feats[key].groupby(tier_name).quantile(0.75) - feats[key].groupby(tier_name).quantile(0.25)).reindex(C.TIER_NAMES)

    # ------------------------------------------------------------------ #
    # Plot 5 - how much variance is TERRAIN vs SESSION-specific? (why orientation is dropped)
    # ------------------------------------------------------------------ #
    series_mean = {c: cube[:, :, i].mean(1) for i, c in enumerate(C.RAW_CHANNELS)}
    for nm in ("gX", "gY", "gZ"):
        series_mean[f"{nm} (tilt)"] = sig[nm].mean(1)
    for nm in ("a_vert__res_std", "a_horiz__mean", "w_mag__mean"):
        series_mean[nm] = feats[nm].to_numpy()
    groups = meta["group_id"].to_numpy()
    tiers = meta["tier"].to_numpy()
    rows = []
    for nm, v in series_mean.items():
        e_t, e_g = _eta_squared(v.astype(np.float64), tiers), _eta_squared(v.astype(np.float64), groups)
        rows.append({"signal": nm, "terrain_share": e_t, "session_only_share": max(e_g - e_t, 0.0), "eta2_session_total": e_g})
    eta = pd.DataFrame(rows).set_index("signal")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    eta[["terrain_share", "session_only_share"]].plot(kind="barh", stacked=True, ax=ax, color=["#2a9d8f", "#bc4749"])
    ax.set_xlabel("share of variance of the per-window mean (eta^2)")
    ax.set_title("Variance explained by terrain tier vs. session-specific effects (TRAIN)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig05_terrain_vs_session_variance.png", dpi=150)
    plt.close(fig)
    out["eta_squared"] = eta

    # ------------------------------------------------------------------ #
    # Plot 6 - PCA of the engineered features: tiers vs sessions
    # ------------------------------------------------------------------ #
    Xs = StandardScaler().fit_transform(np.clip(feats.to_numpy(), *np.percentile(feats.to_numpy(), [0.5, 99.5], axis=0)))
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    Z = pca.transform(Xs)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for t, name in enumerate(C.TIER_NAMES):
        m = tiers == t
        ax[0].scatter(Z[m, 0], Z[m, 1], s=7, alpha=0.55, c=TIER_COLORS[t], label=name)
    ax[0].legend(markerscale=2)
    ax[0].set_title("PCA coloured by risk tier")
    cmap = plt.get_cmap("tab20")
    gid = pd.factorize(groups)[0]
    ax[1].scatter(Z[:, 0], Z[:, 1], s=7, alpha=0.6, c=[cmap(i % 20) for i in gid])
    ax[1].set_title(f"PCA coloured by recording session ({len(np.unique(groups))} sessions)")
    for a in ax:
        a.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var.)")
        a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var.)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig06_pca_tiers_vs_sessions.png", dpi=150)
    plt.close(fig)
    out["pca_var_ratio"] = [float(v) for v in pca.explained_variance_ratio_]
    return out
