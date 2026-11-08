"""
The Overfitting Rover  -  an interactive game that teaches the Bias-Variance Tradeoff.

Course : CS3006 Introduction to Data Science for Engineers
Authors: Sanat (EC24B1069) and Rahul Ambawat (EC24B1088)

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
X_MIN, X_MAX = 0.0, 6.0          # length of the obstacle course (km)
MAX_DEGREE = 15                  # complexity slider range: 1 .. 15
MAX_DEPLOYMENTS = 6              # the player's "fuel": limited deployments
WIN_TOLERANCE = 1.20             # win if test MSE <= 1.20 x best achievable test MSE
CRASH_FACTOR = 2.0               # a test waypoint is a "crash" if error > 2 x noise

DIFFICULTY = {
    #           noise std, n_train, n_test
    "Easy":   (0.25, 25, 40),
    "Normal": (0.35, 20, 40),
    "Hard":   (0.50, 16, 40),
}

st.set_page_config(page_title="The Overfitting Rover", page_icon="🚙", layout="wide")


# ----------------------------------------------------------------------------
# Core computation (real ML, no hard-coded answers)
# ----------------------------------------------------------------------------
def true_terrain(x):
    """The hidden 'real' terrain the rover must learn (unknown to the player)."""
    return 1.5 * np.sin(x) + 0.3 * x


@st.cache_data(show_spinner=False)
def generate_data(seed: int, noise: float, n_train: int, n_test: int):
    """Synthetic waypoints = true terrain + random sensor noise."""
    rng = np.random.default_rng(seed)
    x_train = np.sort(rng.uniform(X_MIN, X_MAX, n_train))
    x_test = np.sort(rng.uniform(X_MIN, X_MAX, n_test))
    y_train = true_terrain(x_train) + rng.normal(0, noise, n_train)
    y_test = true_terrain(x_test) + rng.normal(0, noise, n_test)
    return x_train, y_train, x_test, y_test


def scale_x(x):
    """Map [0, 6] -> [-1, 1] so high-degree polynomials stay numerically stable."""
    return ((np.asarray(x) - (X_MIN + X_MAX) / 2) / ((X_MAX - X_MIN) / 2)).reshape(-1, 1)


def fit_rover_model(x_train, y_train, degree):
    """Fit a polynomial regression of the chosen degree (the 'rover brain')."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    model = LinearRegression()
    model.fit(poly.fit_transform(scale_x(x_train)), y_train)
    return poly, model


def predict_path(poly, model, x):
    return model.predict(poly.transform(scale_x(x)))


def evaluate(degree, data):
    """Return (poly, model, train_mse, test_mse) for a given degree."""
    x_tr, y_tr, x_te, y_te = data
    poly, model = fit_rover_model(x_tr, y_tr, degree)
    train_mse = mean_squared_error(y_tr, predict_path(poly, model, x_tr))
    test_mse = mean_squared_error(y_te, predict_path(poly, model, x_te))
    return poly, model, train_mse, test_mse


@st.cache_data(show_spinner=False)
def error_curve(data_key):
    """Train/test MSE for every degree (used for the hidden target + the reveal)."""
    data = generate_data(*data_key)
    rows = []
    for d in range(1, MAX_DEGREE + 1):
        _, _, tr, te = evaluate(d, data)
        rows.append({"Degree": d, "Train MSE": tr, "Test MSE": te})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def reset_mission(new_seed=False):
    if new_seed or "seed" not in st.session_state:
        st.session_state.seed = int(np.random.default_rng().integers(1, 10_000))
    st.session_state.log = []          # list of deployment dicts
    st.session_state.won = False


if "seed" not in st.session_state:
    reset_mission(new_seed=True)

# ----------------------------------------------------------------------------
# Sidebar: learning objectives + settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎯 Learning Objectives")
    st.markdown(
        """
By the end of this mission you will be able to:

1. **Explain** underfitting (high bias) and overfitting (high variance) in your own words.
2. **Recognise** each one from a plot *and* from Train vs Test error.
3. **Explain** why Train Error alone is a misleading measure of quality.
4. **Find** the sweet spot of model complexity that generalises best.
"""
    )
    st.divider()
    st.header("⚙️ Mission Settings")
    difficulty = st.selectbox("Terrain difficulty (sensor noise)", list(DIFFICULTY), index=1)
    if st.button("🗺️ Generate new terrain", use_container_width=True):
        reset_mission(new_seed=True)
        st.rerun()
    if st.button("🔄 Restart mission (same terrain)", use_container_width=True):
        reset_mission()
        st.rerun()

noise, n_train, n_test = DIFFICULTY[difficulty]
data_key = (st.session_state.seed, noise, n_train, n_test)
data = generate_data(*data_key)
x_tr, y_tr, x_te, y_te = data

curve = error_curve(data_key)
best_test_mse = curve["Test MSE"].min()
target_mse = best_test_mse * WIN_TOLERANCE
crash_tol = CRASH_FACTOR * noise

# ----------------------------------------------------------------------------
# Header + briefing
# ----------------------------------------------------------------------------
st.title("🚙 The Overfitting Rover")
st.caption("Teaching the Bias–Variance Tradeoff  |  CS3006 Introduction to Data Science for Engineers")

with st.expander("📋 Mission Briefing (read first)", expanded=not st.session_state.log):
    st.markdown(
        f"""
You are the **systems engineer** of a Mars rover. The rover's autopilot predicts the
terrain height along a {X_MAX:.0f} km course using a **polynomial model**. You can only
tune one thing: **Model Complexity (polynomial degree)**.

* 🔵 **Blue dots – Training waypoints**: surveyed by the orbiter. The model *learns* from these.
* 🟠 **Orange/green/red markers – Test terrain**: ground the rover has *never seen*. This is the real drive.
* 🟣 **Purple line – the rover's predicted path.**

**Your mission:** drag the slider and *watch the path change*. When you think you have
found the best complexity, press **🚀 Deploy Rover**. You win when the rover's
**Test Error ≤ {target_mse:.3f}** (mean squared error). You have only
**{MAX_DEPLOYMENTS} deployments**, so experiment with the slider first (it is free!).
"""
    )

# ----------------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    degree = st.slider(
        "🎛️ Model Complexity (Polynomial Degree)",
        min_value=1,
        max_value=MAX_DEGREE,
        value=1,
        help="Degree 1 = straight line. Higher degree = more flexible, wigglier path.",
    )

poly, model, train_mse, test_mse = evaluate(degree, data)
train_rmse, test_rmse = np.sqrt(train_mse), np.sqrt(test_mse)

# Which test waypoints would the rover crash on?
test_pred = predict_path(poly, model, x_te)
test_err = np.abs(test_pred - y_te)
crashed = test_err > crash_tol
n_crashes = int(crashed.sum())

with left:
    # ---- Plot ---------------------------------------------------------------
    x_grid = np.linspace(X_MIN, X_MAX, 500)
    y_path = predict_path(poly, model, x_grid)

    all_y = np.concatenate([y_tr, y_te])
    pad = 0.15 * (all_y.max() - all_y.min())
    y_lo, y_hi = all_y.min() - pad, all_y.max() + pad

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(x_grid, true_terrain(x_grid), color="lightgray", lw=6, alpha=0.6,
            label="Real terrain (hidden from rover)", zorder=1)
    ax.plot(x_grid, y_path, color="#7b2cbf", lw=2.6, label=f"Rover path (degree {degree})", zorder=3)
    ax.scatter(x_tr, y_tr, color="#1f77b4", s=55, edgecolor="white", zorder=4,
               label="Training waypoints (seen)")
    ax.scatter(x_te[~crashed], y_te[~crashed], color="#2ca02c", marker="s", s=38, zorder=4,
               label="Test terrain – safe")
    ax.scatter(x_te[crashed], y_te[crashed], color="#d62728", marker="X", s=80, zorder=5,
               label="Test terrain – CRASH")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xlabel("Distance along course (km)")
    ax.set_ylabel("Terrain height (m)")
    ax.set_title("Rover trajectory vs. terrain")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    # Warn when the path leaves the visible window
    if (y_path.max() > y_hi) or (y_path.min() < y_lo):
        ax.text(0.5, 0.03, "⚠ path leaves the chart – rover is flying off course!",
                transform=ax.transAxes, ha="center", color="#d62728", fontsize=10,
                fontweight="bold", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    st.pyplot(fig)
    plt.close(fig)

with right:
    # ---- Metrics ------------------------------------------------------------
    st.subheader("📊 Live Telemetry")
    m1, m2 = st.columns(2)
    m1.metric("Train Error (MSE)", f"{train_mse:.3f}")
    m2.metric("Test Error (MSE)", f"{test_mse:.3f}",
              delta=f"{test_mse - train_mse:+.3f} vs train", delta_color="inverse")
    m3, m4 = st.columns(2)
    m3.metric("Train RMSE (m)", f"{train_rmse:.2f}")
    m4.metric("Test RMSE (m)", f"{test_rmse:.2f}")
    st.metric("💥 Crashes on unseen terrain", f"{n_crashes} / {len(x_te)}")

    st.progress(min(1.0, target_mse / max(test_mse, 1e-9)),
                text=f"Distance to goal  (target Test MSE ≤ {target_mse:.3f})")

    # ---- Diagnosis ----------------------------------------------------------
    st.subheader("🩺 Diagnosis")
    noise_var = noise ** 2
    if test_mse <= target_mse:
        st.success("✅ **Good generalisation.** Train and Test error are both low and close "
                   "together. This model is flexible enough to follow the real terrain without "
                   "memorising noise.")
    elif train_mse > 1.4 * noise_var:
        # Even on the data it was trained on, the model cannot get down to the
        # noise floor -> it is too rigid to follow the real pattern.
        st.warning("📉 **Underfitting (high bias).** The path is too rigid to follow the terrain. "
                   "Even the *training* error is high – the model is *too simple*. "
                   "Increase complexity.")
    elif test_mse > 2.0 * train_mse:
        st.error("📈 **Overfitting (high variance).** Train error is small but Test error is far "
                 "larger. The path bends to hit every training dot, noise included, and fails on "
                 "new ground. The model is *too complex* – reduce complexity.")
    else:
        st.info("🔎 **Getting close.** Nudge the slider by one or two steps and compare the "
                "Test error.")

    # ---- Deploy -------------------------------------------------------------
    st.subheader("🚀 Deployment")
    used = len(st.session_state.log)
    remaining = MAX_DEPLOYMENTS - used
    st.write(f"Deployments left: **{remaining} / {MAX_DEPLOYMENTS}**")

    game_over = st.session_state.won or remaining <= 0
    if st.button("🚀 Deploy Rover at this complexity", type="primary",
                 use_container_width=True, disabled=game_over):
        success = bool(test_mse <= target_mse)
        st.session_state.log.append({
            "Attempt": used + 1,
            "Degree": degree,
            "Train MSE": round(train_mse, 3),
            "Test MSE": round(test_mse, 3),
            "Crashes": n_crashes,
            "Result": "✅ Success" if success else "💥 Failed",
        })
        if success:
            st.session_state.won = True
        st.rerun()

# ----------------------------------------------------------------------------
# Outcome + discovery log
# ----------------------------------------------------------------------------
st.divider()
last = st.session_state.log[-1] if st.session_state.log else None

if st.session_state.won:
    st.balloons()
    st.success(
        f"🏆 **MISSION COMPLETE!** The rover crossed the terrain safely at degree "
        f"**{last['Degree']}**. Review the *Lessons Learned* below."
    )
elif remaining <= 0:
    st.error("⛽ **Mission failed – out of deployments.** Study the error curve below, then "
             "restart or generate a new terrain.")
elif last is not None:
    st.warning(f"💥 Deployment {last['Attempt']} failed at degree {last['Degree']} "
               f"(Test MSE {last['Test MSE']} > target {target_mse:.3f}). Read the "
               "diagnosis and try again.")

if st.session_state.log:
    st.subheader("📓 Engineer's Log – your experiments")
    log_df = pd.DataFrame(st.session_state.log)
    c1, c2 = st.columns([2, 3], gap="large")
    c1.dataframe(log_df, hide_index=True, use_container_width=True)

    # Player's own discovered error curve
    fig2, ax2 = plt.subplots(figsize=(6, 3.4))
    pts = log_df.sort_values("Degree")
    ax2.plot(pts["Degree"], pts["Train MSE"], "o-", color="#1f77b4", label="Train error")
    ax2.plot(pts["Degree"], pts["Test MSE"], "s-", color="#d62728", label="Test error")
    ax2.axhline(target_mse, color="green", ls="--", lw=1, label="Win threshold")
    ax2.set_xlabel("Model complexity (degree)")
    ax2.set_ylabel("MSE")
    ax2.set_title("Errors from your deployments")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)
    c2.pyplot(fig2)
    plt.close(fig2)

# Reveal full bias-variance curve only after the game ends (discovery first, theory second)
if st.session_state.won or remaining <= 0:
    st.subheader("🎓 Lessons Learned – the full Bias–Variance picture")
    best_deg = int(curve.loc[curve["Test MSE"].idxmin(), "Degree"])
    fig3, ax3 = plt.subplots(figsize=(8, 3.8))
    ax3.plot(curve["Degree"], curve["Train MSE"], "o-", color="#1f77b4", label="Train error")
    ax3.plot(curve["Degree"], curve["Test MSE"], "s-", color="#d62728", label="Test error")
    ax3.axvline(best_deg, color="green", ls="--", label=f"Lowest test error (degree {best_deg})")
    ax3.axvspan(0.5, max(1.5, best_deg - 1.5), color="orange", alpha=0.10, label="Underfitting zone")
    ax3.axvspan(best_deg + 2.5, MAX_DEGREE + 0.5, color="red", alpha=0.08, label="Overfitting zone")
    ax3.set_yscale("log")
    ax3.set_xlabel("Model complexity (degree)")
    ax3.set_ylabel("MSE (log scale)")
    ax3.set_title("Train error keeps falling – Test error is U-shaped")
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=8, ncol=2)
    st.pyplot(fig3)
    plt.close(fig3)
    st.markdown(
        """
* **Underfitting (high bias):** the model is too simple, so it misses the real pattern. *Both* errors are high.
* **Overfitting (high variance):** the model is so flexible it memorises noise in the training data. Train error is tiny but Test error explodes.
* **The tradeoff:** as complexity grows, bias falls but variance rises. Test error is lowest in the middle – the **sweet spot**.
* **Key lesson:** *a low Train Error does not mean a good model* – only performance on unseen data counts.
"""
    )
