# Terrain-risk classification for autonomous rovers (IMU time series)

End-to-end, reproducible pipeline: real robot IMU data -> leakage-free session split -> EDA ->
Logistic Regression baseline, Random Forest and a PyTorch 1D-CNN -> risk-aware evaluation and error analysis.

**The full write-up (for presentation and viva) is `REPORT.md`.**

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # exact tested versions: requirements-tested.txt
python run_pipeline.py                                 # full run, ~15 min on one CPU core
python run_pipeline.py --quick                         # ~2 min smoke test (numbers NOT for the report)
```

The dataset (Kaggle *CareerCon 2019 - Help Navigate Robots*, `X_train.csv` + `y_train.csv`, ~49 MB) is
downloaded automatically into `data/` and verified with SHA-256. If your network blocks GitHub, download the
two files from <https://www.kaggle.com/c/career-con-2019/data> and put them in `data/`.

PyTorch tip (CPU only, much smaller download on Linux/Windows):
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Layout

| File | Purpose |
|---|---|
| `config.py` | every constant: seed, paths, surface->tier mapping, split sizes, project goals |
| `data.py` | download, audit/cleaning (types, missing, duplicates, outliers), session-level splits |
| `features.py` | gravity/tilt signals, rolling-mean spike features, spectral features, train-fitted scalers |
| `eda.py` | six EDA figures (TRAIN sessions only) |
| `models.py` | seeds, Logistic Regression, Random Forest, PyTorch CNN, grouped-CV tuning |
| `evaluate.py` | risk-aware metrics, session bootstrap, safety threshold, error analysis, plots |
| `run_pipeline.py` | orchestrates everything and writes `outputs/` |
| `outputs/figures/` | all plots (fig01 ... fig12) |
| `outputs/results/` | CSV/JSON results and `results_summary.md` (machine-written numbers behind the report) |

## Reproducibility

`models.set_global_seed(42)` seeds Python, NumPy and PyTorch, enables deterministic PyTorch algorithms, and every
sklearn estimator / splitter receives `random_state`. Results were produced on CPU with the versions in
`requirements-tested.txt`; other hardware or library versions can change the last decimals (especially the CNN).
