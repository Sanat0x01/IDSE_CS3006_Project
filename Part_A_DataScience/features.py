"""
features.py -- physics-informed signals, hand-crafted window features and the
train-fitted transformers used by the models.

Key engineering decisions (all justified in the report)
-------------------------------------------------------
* The raw orientation quaternion is NOT used directly. It contains the robot's
  absolute heading (yaw), which depends on which way the robot happened to face in
  a given recording session -> it identifies the *session*, not the *terrain*.
  Instead we convert it to the direction of gravity in the sensor frame
  (tilt only, heading-free).
* Vertical vs horizontal acceleration is split with that gravity direction:
  vertical = vibration / bumps, horizontal = traction, drag, slip.
* A centred rolling mean separates the slow motion of the robot from high-frequency
  spikes (residual = signal - rolling mean).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, TransformerMixin

import config as C

# indices into C.RAW_CHANNELS
_Q = slice(0, 4)      # orientation x, y, z, w
_W = slice(4, 7)      # angular velocity x, y, z   (rad/s)
_A = slice(7, 10)     # linear acceleration x, y, z (m/s^2, includes gravity)

SIGNAL_NAMES = ["wX", "wY", "wZ", "aX", "aY", "aZ", "gX", "gY", "gZ",
                "a_vert", "a_horiz", "a_mag", "w_mag"]
SEQ_SIGNALS = ["wX", "wY", "wZ", "aX", "aY", "aZ", "gX", "gY", "gZ", "a_vert", "a_horiz"]


# --------------------------------------------------------------------------- #
# 1. Physical signal derivation
# --------------------------------------------------------------------------- #
def derive_signals(cube: np.ndarray) -> dict[str, np.ndarray]:
    """cube (N,128,10) -> dict of (N,128) arrays named as in SIGNAL_NAMES."""
    q = cube[:, :, _Q].astype(np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=2, keepdims=True), 1e-9, None)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Direction of the world "up" axis expressed in the sensor frame
    # (3rd row of the body->world rotation matrix). Independent of yaw/heading.
    gX = 2 * (x * z - w * y)
    gY = 2 * (y * z + w * x)
    gZ = 1 - 2 * (x * x + y * y)

    w3 = cube[:, :, _W].astype(np.float64)
    a3 = cube[:, :, _A].astype(np.float64)
    a_vert = a3[..., 0] * gX + a3[..., 1] * gY + a3[..., 2] * gZ      # component along "up"
    a_mag = np.linalg.norm(a3, axis=2)
    a_horiz = np.sqrt(np.clip(a_mag ** 2 - a_vert ** 2, 0, None))     # in-plane component
    w_mag = np.linalg.norm(w3, axis=2)

    return {"wX": w3[..., 0], "wY": w3[..., 1], "wZ": w3[..., 2],
            "aX": a3[..., 0], "aY": a3[..., 1], "aZ": a3[..., 2],
            "gX": gX, "gY": gY, "gZ": gZ,
            "a_vert": a_vert, "a_horiz": a_horiz, "a_mag": a_mag, "w_mag": w_mag}


def rolling_residual(sig: np.ndarray, window: int = C.ROLL_WINDOW) -> np.ndarray:
    """signal - centred rolling mean (same as pandas rolling(window, center=True).mean())."""
    return sig - uniform_filter1d(sig, size=window, axis=1, mode="nearest")


# --------------------------------------------------------------------------- #
# 2. Window-level feature table (for Logistic Regression / Random Forest)
# --------------------------------------------------------------------------- #
def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-12
    return (a * b).sum(1) / den


def _spectral(sig: np.ndarray, n_bands: int = C.N_SPECTRAL_BANDS):
    """Band-power fractions, dominant-frequency position and spectral entropy.
    Frequencies are normalised (fraction of Nyquist) because the sampling rate
    is not documented in the dataset."""
    s = sig - sig.mean(1, keepdims=True)
    s = s * np.hanning(s.shape[1])[None, :]
    p = np.abs(np.fft.rfft(s, axis=1)) ** 2
    p = p[:, 1:]                                                     # drop DC
    tot = p.sum(1, keepdims=True) + 1e-12
    edges = np.linspace(0, p.shape[1], n_bands + 1).astype(int)
    bands = np.stack([p[:, edges[i]:edges[i + 1]].sum(1) for i in range(n_bands)], axis=1) / tot
    pn = p / tot
    entropy = -(pn * np.log(pn + 1e-12)).sum(1) / np.log(p.shape[1])
    dom = p.argmax(1) / p.shape[1]
    return bands, dom, entropy


def extract_features(cube: np.ndarray) -> pd.DataFrame:
    """One row of ~150 engineered features per 128-sample window."""
    sig = derive_signals(cube)
    f: dict[str, np.ndarray] = {}

    dynamic = [s for s in SIGNAL_NAMES if s not in ("gX", "gY", "gZ")]
    for name in dynamic:
        s = sig[name]
        d = np.diff(s, axis=1)
        f[f"{name}__mean"] = s.mean(1)
        f[f"{name}__std"] = s.std(1)
        f[f"{name}__min"] = s.min(1)
        f[f"{name}__max"] = s.max(1)
        f[f"{name}__ptp"] = np.ptp(s, axis=1)
        f[f"{name}__skew"] = skew(s, axis=1)
        f[f"{name}__kurt"] = kurtosis(s, axis=1)
        f[f"{name}__mad_diff"] = np.abs(d).mean(1)            # jerk proxy

        # ---- rolling-mean spike features -------------------------------- #
        r = rolling_residual(s)
        sigma = 1.4826 * np.median(np.abs(r - np.median(r, 1, keepdims=True)), 1) + 1e-9
        f[f"{name}__res_std"] = r.std(1)
        f[f"{name}__res_maxabs"] = np.abs(r).max(1)
        f[f"{name}__spike_frac"] = (np.abs(r) > C.SPIKE_SIGMA * sigma[:, None]).mean(1)

    for name in ("gX", "gY", "gZ"):                            # tilt: only level + wobble
        f[f"{name}__mean"] = sig[name].mean(1)
        f[f"{name}__std"] = sig[name].std(1)

    for name in ("a_vert", "a_horiz", "wX", "wY", "wZ"):       # frequency content
        bands, dom, ent = _spectral(sig[name])
        for i in range(bands.shape[1]):
            f[f"{name}__band{i}"] = bands[:, i]
        f[f"{name}__dom_freq"] = dom
        f[f"{name}__spec_entropy"] = ent

    f["corr__avert_wX"] = _corr(sig["a_vert"], sig["wX"])      # bump <-> pitch/roll coupling
    f["corr__avert_wY"] = _corr(sig["a_vert"], sig["wY"])
    f["corr__aX_aY"] = _corr(sig["aX"], sig["aY"])

    df = pd.DataFrame(f).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. Raw-sequence tensor for the CNN
# --------------------------------------------------------------------------- #
def build_sequences(cube: np.ndarray) -> np.ndarray:
    """(N,128,10) -> (N, n_channels, 128) float32, ready for Conv1d."""
    sig = derive_signals(cube)
    return np.stack([sig[n] for n in SEQ_SIGNALS], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 4. Transformers that are FITTED ON TRAIN ONLY (sklearn-style fit/transform)
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip every feature to [q_low, q_high] percentiles learnt on the training data.
    Tames extreme outliers (important for Logistic Regression) without removing rows."""

    def __init__(self, q_low: float = 0.5, q_high: float = 99.5):
        self.q_low, self.q_high = q_low, q_high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.lo_, self.hi_ = np.percentile(X, [self.q_low, self.q_high], axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=np.float64), self.lo_, self.hi_)


class SequenceScaler:
    """Per-channel winsorise + z-score for the CNN input, fitted on TRAIN windows only."""

    def __init__(self, q_low: float = 0.1, q_high: float = 99.9):
        self.q_low, self.q_high = q_low, q_high

    def fit(self, seq: np.ndarray):                      # seq: (N, C, T)
        flat = seq.transpose(1, 0, 2).reshape(seq.shape[1], -1)
        self.lo_, self.hi_ = (v[:, None] for v in np.percentile(flat, [self.q_low, self.q_high], axis=1))
        clipped = np.clip(flat, self.lo_, self.hi_)
        self.mu_, self.sd_ = clipped.mean(1, keepdims=True), clipped.std(1, keepdims=True) + 1e-8
        return self

    def transform(self, seq: np.ndarray) -> np.ndarray:
        out = np.empty_like(seq, dtype=np.float32)
        for c in range(seq.shape[1]):
            out[:, c] = (np.clip(seq[:, c], self.lo_[c, 0], self.hi_[c, 0]) - self.mu_[c, 0]) / self.sd_[c, 0]
        return out
