"""
models.py -- baseline, Random Forest, PyTorch 1D-CNN and the tuning routines.

Tuning protocol (identical for every model)
-------------------------------------------
* Hyper-parameters are chosen with STRATIFIED GROUP k-fold CV on the TRAIN
  sessions only (folds never split a session). Validation / test sessions are
  never seen during tuning.
* Selection metric = macro-F1 (balanced over the three tiers); hazard recall is
  logged next to it so that a candidate that "wins" by missing hazards is visible.
* All fitted pre-processing (winsorising, scaling) lives INSIDE the CV loop /
  sklearn Pipeline, i.e. it is re-fitted on each fold's training part only.
"""
from __future__ import annotations

import copy
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from features import SequenceScaler, Winsorizer


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int = C.SEED) -> None:
    """Fix every random number generator that can influence the results."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- #
# Metric helpers shared by tuning and evaluation
# --------------------------------------------------------------------------- #
def hazard_recall_score(y_true, y_pred) -> float:
    """Of all truly hazardous windows (tier >= 1), the share predicted as hazardous."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pos = y_true >= 1
    return float(((y_pred >= 1) & pos).sum() / max(pos.sum(), 1))


SCORERS = {"f1_macro": "f1_macro", "hazard_recall": make_scorer(hazard_recall_score)}


# --------------------------------------------------------------------------- #
# Classical models on the engineered feature table
# --------------------------------------------------------------------------- #
def build_dummy() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def build_logreg() -> Pipeline:
    """Baseline: winsorise -> standardise -> multinomial L2 logistic regression."""
    return Pipeline([
        ("win", Winsorizer()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=C.SEED)),
    ])


def build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=C.SEED)


LOGREG_GRID = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]}
RF_SPACE = {
    "n_estimators": [200, 400],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 3, 8, 15],      # larger leaves = smoother, less session-specific trees
    "max_features": ["sqrt", 0.2, 0.4],
}


def tune_sklearn(estimator, space: dict, X, y, folds, kind: str = "grid", n_iter: int = 12):
    """Grid (small, exhaustive) or randomised search with the pre-computed grouped folds."""
    common = dict(scoring=SCORERS, refit="f1_macro", cv=folds, n_jobs=1, return_train_score=False)
    if kind == "grid":
        search = GridSearchCV(estimator, space, **common)
    else:
        search = RandomizedSearchCV(estimator, space, n_iter=n_iter, random_state=C.SEED, **common)
    search.fit(X, y)
    res = pd.DataFrame(search.cv_results_)
    keep = ["params", "mean_test_f1_macro", "std_test_f1_macro", "mean_test_hazard_recall", "rank_test_f1_macro"]
    return search.best_estimator_, search.best_params_, res[keep].sort_values("rank_test_f1_macro")


# --------------------------------------------------------------------------- #
# PyTorch 1D-CNN for the raw IMU windows
# --------------------------------------------------------------------------- #
class _CNN(nn.Module):
    """3 x (Conv1d-BN-ReLU) -> global average + max pooling -> dropout -> linear."""

    def __init__(self, n_in: int, n_classes: int, width: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(n_in, width, kernel_size=7, padding=3), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, 2 * width, kernel_size=5, padding=2), nn.BatchNorm1d(2 * width), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(2 * width, 2 * width, kernel_size=3, padding=1), nn.BatchNorm1d(2 * width), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width, n_classes))

    def forward(self, x):
        h = self.body(x)
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)   # global avg + max pooling
        return self.head(h)


class TorchTerrainCNN:
    """Small sklearn-like wrapper (fit / predict_proba / predict) around the CNN.

    * class-weighted cross-entropy (the tiers are imbalanced),
    * AdamW + cosine learning-rate decay + gradient clipping,
    * light Gaussian-noise augmentation (sensor noise robustness),
    * optional validation set: the checkpoint with the best validation macro-F1 is kept.
    """

    def __init__(self, n_in: int, n_classes: int = 3, width: int = 32, dropout: float = 0.3,
                 lr: float = 3e-3, weight_decay: float = 1e-3, epochs: int = 30,
                 batch_size: int = 64, noise: float = 0.05, seed: int = C.SEED):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        set_global_seed(self.seed)
        self.model = _CNN(self.n_in, self.n_classes, self.width, self.dropout)
        Xt, yt = torch.from_numpy(X).float(), torch.from_numpy(y).long()
        counts = np.bincount(y, minlength=self.n_classes)
        weights = torch.tensor(len(y) / (self.n_classes * np.maximum(counts, 1)), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        gen = torch.Generator().manual_seed(self.seed)

        self.history_, best_state, best_key = [], None, (-1.0, 0.0)
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(Xt), generator=gen)
            running, seen = 0.0, 0
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:                       # BatchNorm needs > 1 sample
                    continue
                xb = Xt[idx]
                if self.noise > 0:
                    xb = xb + self.noise * torch.randn(xb.shape, generator=gen)
                opt.zero_grad()
                loss = criterion(self.model(xb), yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                running += loss.item() * len(idx)
                seen += len(idx)
            sched.step()

            rec = {"epoch": epoch + 1, "train_loss": running / max(seen, 1)}
            if X_val is not None:
                proba = self.predict_proba(X_val)
                rec["val_macro_f1"] = f1_score(y_val, proba.argmax(1), average="macro")
                rec["val_loss"] = log_loss(y_val, proba, labels=list(range(self.n_classes)))
                key = (rec["val_macro_f1"], -rec["val_loss"])
                if key > best_key:                     # keep the best checkpoint
                    best_key, best_state, rec["is_best"] = key, copy.deepcopy(self.model.state_dict()), True
            self.history_.append(rec)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = []
        for i in range(0, len(X), 512):
            logits = self.model(torch.from_numpy(X[i:i + 512]).float())
            out.append(torch.softmax(logits, dim=1).numpy())
        return np.concatenate(out)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


CNN_GRID = [{"width": w, "dropout": d, "lr": lr}
            for w in (16, 32) for d in (0.3, 0.5) for lr in (1e-3, 3e-3)]


def tune_cnn(seq: np.ndarray, y: np.ndarray, folds, grid=CNN_GRID, epochs: int = 30, verbose: bool = True):
    """Grouped-CV grid search for the CNN. A fixed training budget (no early stopping on the
    fold's validation part) keeps the CV estimate free of selection bias; the scaler is
    re-fitted on every fold's training part."""
    rows = []
    for cfg in grid:
        f1s, recs = [], []
        t0 = time.time()
        for tr, va in folds:
            scaler = SequenceScaler().fit(seq[tr])
            net = TorchTerrainCNN(n_in=seq.shape[1], epochs=epochs, **cfg)
            net.fit(scaler.transform(seq[tr]), y[tr])
            pred = net.predict(scaler.transform(seq[va]))
            f1s.append(f1_score(y[va], pred, average="macro"))
            recs.append(hazard_recall_score(y[va], pred))
        rows.append({"params": cfg, "mean_test_f1_macro": np.mean(f1s), "std_test_f1_macro": np.std(f1s),
                     "mean_test_hazard_recall": np.mean(recs)})
        if verbose:
            print(f"   CNN {cfg}: macro-F1 {np.mean(f1s):.3f} +/- {np.std(f1s):.3f} "
                  f"hazard-recall {np.mean(recs):.3f}  ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    res["rank_test_f1_macro"] = res["mean_test_f1_macro"].rank(ascending=False, method="first").astype(int)
    res = res.sort_values("rank_test_f1_macro")
    return res.iloc[0]["params"], res
