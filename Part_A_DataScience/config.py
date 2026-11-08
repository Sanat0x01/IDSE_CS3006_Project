"""
config.py -- single source of truth for every setting used by the pipeline.

Keeping all constants here means (a) nothing is "hard-coded" deep inside the
code, and (b) a viva examiner can see every modelling assumption in one place.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"

# --------------------------------------------------------------------------- #
# Dataset: Kaggle "CareerCon 2019 - Help Navigate Robots" (robot IMU time series)
# The competition files are re-hosted in a public GitHub repository, which lets
# the pipeline download them without a Kaggle account. SHA-256 hashes make sure
# you get byte-identical files (and therefore identical results).
# If the download is blocked, place X_train.csv and y_train.csv into ./data/
# (from https://www.kaggle.com/c/career-con-2019/data) and re-run.
# --------------------------------------------------------------------------- #
MIRROR_BASE_URL = "https://raw.githubusercontent.com/gonultasbu/kaggle_careercon_2019/master"
DATA_FILES = {
    "X_train.csv": "3832255baaaf8ab28223ddd1187f6fcfd1ba9912f6cf3e71571d50003de4be76",
    "y_train.csv": "4f8b557b368d8b4111b5497a1055698943627f65577ac0b2fe91cbc76b6961fe",
}

N_STEPS = 128  # every series has exactly 128 consecutive IMU samples
RAW_CHANNELS = [
    "orientation_X", "orientation_Y", "orientation_Z", "orientation_W",
    "angular_velocity_X", "angular_velocity_Y", "angular_velocity_Z",
    "linear_acceleration_X", "linear_acceleration_Y", "linear_acceleration_Z",
]

# --------------------------------------------------------------------------- #
# Problem definition: floor surface -> navigation-risk tier
# --------------------------------------------------------------------------- #
# The dataset labels the *material* the robot drives on (9 classes). It contains
# no stall events, so "terrain stability" is operationalised through an explicit,
# physically-motivated grouping of the materials into 3 navigation-risk tiers.
# This is an ASSUMPTION (documented in the report) and can be edited here.
TIER_NAMES = ["Firm", "Compliant", "Irregular"]  # tier ids 0, 1, 2
SURFACE_TO_TIER = {
    # Tier 0 - firm, low rolling resistance, smooth: safe to drive at full speed
    "concrete": 0, "fine_concrete": 0, "tiled": 0, "hard_tiles": 0, "wood": 0,
    # Tier 1 - compliant / high drag: wheel sinkage + slip -> stall risk
    "carpet": 1, "soft_pvc": 1, "soft_tiles": 1,
    # Tier 2 - irregular (tile joints / large gaps): jolts, loss of traction
    "hard_tiles_large_space": 2,
}
HAZARD_TIERS = (1, 2)  # tiers that require speed / torque adaptation

# --------------------------------------------------------------------------- #
# Splitting (always by recording session = group_id, never by single series)
# --------------------------------------------------------------------------- #
TEST_FRAC = 0.20          # ~20 % of sessions are locked away until the very end
VAL_FRAC = 0.20           # ~20 % of sessions for early stopping / model selection
CV_FOLDS = 3              # grouped CV folds used for tuning on the TRAIN sessions
ROBUST_CV_FOLDS = 5       # outer grouped CV over ALL sessions (stability check)

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
ROLL_WINDOW = 5           # centred rolling-mean window used to isolate spikes
SPIKE_SIGMA = 3.0         # a residual > SPIKE_SIGMA * robust-sigma counts as a spike
N_SPECTRAL_BANDS = 4

# --------------------------------------------------------------------------- #
# Measurable project goals (fixed BEFORE running the experiments)
# --------------------------------------------------------------------------- #
GOALS = {
    "hazard_recall": 0.90,   # >= 90 % of compliant/irregular windows must be flagged
    "macro_f1": 0.80,        # balanced quality over the three tiers
}
SAFETY_RECALL_TARGET = 0.95  # target used when tuning the safety decision threshold
