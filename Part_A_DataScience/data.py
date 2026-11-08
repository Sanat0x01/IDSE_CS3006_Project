"""
data.py -- dataset download, loading, audit/cleaning and leakage-free splitting.

Design rule for zero data leakage
---------------------------------
Everything in this file is either
  (a) label-free and row-wise (type casting, duplicate removal, interpolation of a
      gap *inside one series*), which cannot leak information between series, or
  (b) a split that keeps whole recording sessions (group_id) together.
Every statistic that is *fitted* (clipping bounds, scalers, ...) lives in
features.py / models.py and is fitted on TRAIN data only.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config as C


# --------------------------------------------------------------------------- #
# 1. Download + load
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dataset(data_dir: Path = C.DATA_DIR) -> None:
    """Download the dataset files if they are not already in `data_dir`."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in C.DATA_FILES.items():
        path = data_dir / name
        if path.exists():
            if _sha256(path) != expected_hash:
                print(f"[data] WARNING: {name} differs from the reference copy "
                      f"(different export?). Continuing with your file.")
            continue
        url = f"{C.MIRROR_BASE_URL}/{name}"
        print(f"[data] downloading {url}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:  # network blocked, etc.
            raise RuntimeError(
                f"Could not download {name} ({exc}).\nPlease download X_train.csv and "
                f"y_train.csv from https://www.kaggle.com/c/career-con-2019/data and "
                f"copy them into {data_dir}, then re-run."
            ) from exc
        if _sha256(path) != expected_hash:
            path.unlink()
            raise RuntimeError(f"Checksum mismatch for {name}; file removed. Re-run to retry.")


def load_raw(data_dir: Path = C.DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the long-format sensor table and the label table."""
    ensure_dataset(data_dir)
    X = pd.read_csv(data_dir / "X_train.csv")
    y = pd.read_csv(data_dir / "y_train.csv")
    return X, y


# --------------------------------------------------------------------------- #
# 2. Audit + cleaning (missing values, duplicates, data types, outliers)
# --------------------------------------------------------------------------- #
def validate_and_clean(X: pd.DataFrame, y: pd.DataFrame, max_missing_frac: float = 0.10):
    """
    Returns
    -------
    cube : float32 array (n_series, 128, 10)   -- sensor windows, channel order = C.RAW_CHANNELS
    meta : DataFrame (n_series)                -- series_id, group_id, surface, tier
    audit: dict                                -- counts of everything that was checked / changed
    """
    audit: dict = {"rows_in": int(len(X)), "series_in": int(y["series_id"].nunique())}

    # --- schema ---------------------------------------------------------- #
    need = {"series_id", "measurement_number", *C.RAW_CHANNELS}
    missing_cols = need - set(X.columns)
    if missing_cols or not {"series_id", "group_id", "surface"} <= set(y.columns):
        raise ValueError(f"Unexpected schema. Missing sensor columns: {missing_cols}")

    # --- data types ------------------------------------------------------ #
    # float32 halves memory (487k x 10 values) and is far more precise than the
    # IMU's own resolution; ids become int32; the target text becomes a string.
    X = X.copy()
    X["series_id"] = X["series_id"].astype("int32")
    X["measurement_number"] = X["measurement_number"].astype("int16")
    X[C.RAW_CHANNELS] = X[C.RAW_CHANNELS].astype("float32")
    y = y.copy()
    y["series_id"] = y["series_id"].astype("int32")
    y["group_id"] = y["group_id"].astype("int32")
    y["surface"] = y["surface"].astype(str).str.strip().str.lower()
    audit["dtypes_after"] = {"sensor": "float32", "ids": "int32", "surface": "str"}

    # --- duplicates ------------------------------------------------------ #
    # (a) an identical (series, time-step) pair twice would silently lengthen a
    #     window and distort every statistic -> keep the first occurrence only.
    audit["duplicate_rows_exact"] = int(X.duplicated().sum())
    audit["duplicate_keys"] = int(X.duplicated(["series_id", "measurement_number"]).sum())
    X = X.drop_duplicates(["series_id", "measurement_number"], keep="first")
    audit["duplicate_label_rows"] = int(y.duplicated("series_id").sum())
    y = y.drop_duplicates("series_id", keep="first")

    # --- missing values -------------------------------------------------- #
    audit["missing_sensor_values"] = int(X[C.RAW_CHANNELS].isna().sum().sum())
    audit["missing_labels"] = int(y[["group_id", "surface"]].isna().any(axis=1).sum())
    y = y.dropna(subset=["group_id", "surface"])            # an unlabeled series is unusable
    X = X[X["series_id"].isin(y["series_id"])]
    if audit["missing_sensor_values"] > 0:
        # Windows with too many gaps are dropped; small gaps are filled by linear
        # interpolation *inside the same series only* (no information crosses series).
        frac = X.groupby("series_id")[C.RAW_CHANNELS].apply(lambda d: d.isna().mean().max())
        bad = frac.index[frac > max_missing_frac]
        audit["series_dropped_missing"] = int(len(bad))
        X = X[~X["series_id"].isin(bad)]
        X = X.sort_values(["series_id", "measurement_number"])
        X[C.RAW_CHANNELS] = (
            X.groupby("series_id")[C.RAW_CHANNELS]
            .transform(lambda s: s.interpolate(limit_direction="both"))
        )
    else:
        audit["series_dropped_missing"] = 0

    # --- completeness: exactly 128 contiguous steps per series ------------- #
    X = X.sort_values(["series_id", "measurement_number"])
    counts = X.groupby("series_id").size()
    complete = counts.index[counts == C.N_STEPS]
    audit["series_dropped_incomplete"] = int((counts != C.N_STEPS).sum())
    X = X[X["series_id"].isin(complete)]
    y = y[y["series_id"].isin(complete)].sort_values("series_id").reset_index(drop=True)
    ids = y["series_id"].to_numpy()
    assert np.array_equal(np.unique(X["series_id"].to_numpy()), ids), "sensor/label series mismatch"
    cube = X[C.RAW_CHANNELS].to_numpy(dtype=np.float32).reshape(len(ids), C.N_STEPS, len(C.RAW_CHANNELS))

    # (b) whole-series duplicates (same 128x10 values) would let a model "see" a
    #     test series during training -> keep the first copy only.
    digests = pd.Series([hashlib.md5(w.tobytes()).hexdigest() for w in cube])
    dup_series = digests.duplicated().to_numpy()
    audit["duplicate_series"] = int(dup_series.sum())
    cube, y = cube[~dup_series], y[~dup_series].reset_index(drop=True)

    # --- quaternion sanity: unit norm ------------------------------------ #
    q = cube[:, :, 0:4]
    norm = np.linalg.norm(q, axis=2)
    audit["quaternion_norm_min"] = float(norm.min())
    audit["quaternion_norm_max"] = float(norm.max())
    audit["quaternions_renormalised"] = int((np.abs(norm - 1) > 1e-3).sum())
    cube[:, :, 0:4] = q / np.clip(norm[..., None], 1e-8, None)

    # --- outliers: AUDIT ONLY (robust z-score via median/MAD) -------------- #
    # We do NOT delete outlying samples: (i) a series must keep its 128 steps,
    # and (ii) a spike on tile joints or carpet is often the very signal we want.
    # Treatment = winsorising with bounds fitted on TRAIN data only (features.py).
    flat = cube.reshape(-1, cube.shape[2])
    med = np.median(flat, axis=0)
    mad = np.median(np.abs(flat - med), axis=0) * 1.4826 + 1e-9
    rz = np.abs((flat - med) / mad)
    audit["outlier_pct_robust_z_gt_6"] = {
        ch: round(float((rz[:, i] > 6).mean() * 100), 3) for i, ch in enumerate(C.RAW_CHANNELS)
    }
    audit["channel_min"] = {ch: float(flat[:, i].min()) for i, ch in enumerate(C.RAW_CHANNELS)}
    audit["channel_max"] = {ch: float(flat[:, i].max()) for i, ch in enumerate(C.RAW_CHANNELS)}

    # --- labels: one surface per session, then map to risk tiers ------------ #
    audit["groups_with_multiple_surfaces"] = int((y.groupby("group_id")["surface"].nunique() > 1).sum())
    unknown = set(y["surface"]) - set(C.SURFACE_TO_TIER)
    if unknown:
        raise ValueError(f"Surfaces without a tier mapping in config.py: {unknown}")
    meta = y[["series_id", "group_id", "surface"]].copy()
    meta["tier"] = meta["surface"].map(C.SURFACE_TO_TIER).astype(int)

    audit.update(series_out=int(len(meta)), groups=int(meta["group_id"].nunique()),
                 rows_out=int(len(meta) * C.N_STEPS))
    return cube, meta, audit


# --------------------------------------------------------------------------- #
# 3. Leakage-free splitting (always by recording session)
# --------------------------------------------------------------------------- #
def _n_groups_per_tier(meta: pd.DataFrame, idx: np.ndarray) -> dict:
    sub = meta.iloc[idx]
    return sub.groupby("tier")["group_id"].nunique().reindex(range(len(C.TIER_NAMES)), fill_value=0).to_dict()


def make_group_splits(meta: pd.DataFrame, seed: int = C.SEED) -> dict:
    """
    60 / 20 / 20 split by SESSION (group_id), stratified by risk tier.

    Why by session?  Series recorded in the same session are consecutive slices of
    one continuous drive on the same floor. If they were shuffled across
    train/test, the model could "recognise the session" (sensor bias, heading,
    local floor wear) instead of learning the terrain -> inflated scores.

    A split is accepted only if it (1) has zero group overlap, (2) has every tier
    in val and test, (3) leaves enough TRAIN sessions per tier for grouped CV.
    The loop only changes the shuffling seed deterministically; it never looks at
    model performance.
    """
    y, g = meta["tier"].to_numpy(), meta["group_id"].to_numpy()
    n_outer = int(round(1 / C.TEST_FRAC))
    n_inner = int(round((1 - C.TEST_FRAC) / C.VAL_FRAC))
    n_tiers = len(C.TIER_NAMES)

    for attempt in range(500):
        rs = seed + attempt
        outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=rs)
        dev_idx, test_idx = next(iter(outer.split(np.zeros(len(y)), y, g)))
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=rs)
        tr_rel, va_rel = next(iter(inner.split(np.zeros(len(dev_idx)), y[dev_idx], g[dev_idx])))
        train_idx, val_idx = dev_idx[tr_rel], dev_idx[va_rel]

        tr_t, va_t, te_t = (_n_groups_per_tier(meta, i) for i in (train_idx, val_idx, test_idx))
        ok = (
            all(va_t[t] >= 1 and te_t[t] >= 1 for t in range(n_tiers))
            and all(tr_t[t] >= C.CV_FOLDS for t in range(n_tiers))
            and 0.15 <= len(test_idx) / len(y) <= 0.25
            and 0.15 <= len(val_idx) / len(y) <= 0.25
        )
        if ok:
            break
    else:
        raise RuntimeError("No valid group split found; relax the constraints in make_group_splits().")

    # ---- hard guarantees against leakage ---------------------------------- #
    gs = {k: set(g[v]) for k, v in dict(train=train_idx, val=val_idx, test=test_idx).items()}
    assert not (gs["train"] & gs["val"]) and not (gs["train"] & gs["test"]) and not (gs["val"] & gs["test"]), \
        "LEAKAGE: a recording session appears in more than one split"
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(meta)

    manifest = (meta.groupby("group_id").agg(surface=("surface", "first"), tier=("tier", "first"),
                                              n_series=("series_id", "size")).reset_index())
    manifest["split"] = manifest["group_id"].map(
        {gid: name for name, s in gs.items() for gid in s})
    return dict(train=train_idx, val=val_idx, test=test_idx, manifest=manifest, attempt=attempt)


def grouped_cv_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int = C.SEED):
    """Stratified GROUP k-fold with a check that every class is in every train/val part."""
    classes = np.unique(y)
    for attempt in range(500):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + attempt)
        folds = list(skf.split(np.zeros(len(y)), y, groups))
        good = all(
            len(np.unique(y[tr])) == len(classes) and len(np.unique(y[va])) == len(classes)
            and not (set(groups[tr]) & set(groups[va]))
            for tr, va in folds
        )
        if good:
            return folds
    raise RuntimeError("Could not build grouped CV folds containing every class.")
