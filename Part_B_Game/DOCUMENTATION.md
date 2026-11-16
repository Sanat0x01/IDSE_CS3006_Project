**Course:** CS3006 Introduction to Data Science for Engineers

**Professor:** Dr. Vijaya Kumar S

**Institute:** IIITDM Kancheepuram

**Authors:** Sanat (EC24B1069), Rahul Ambawat (EC24B1088) and Vishwajeet Kumar (EC24I2019)

---

# 🚙 The Overfitting Rover — Teaching the Bias–Variance Tradeoff

## 1. Overview

*The Overfitting Rover* is an interactive educational game built with **Streamlit**. The player is the systems engineer of a rover whose autopilot predicts terrain height with a **polynomial regression model**. The only control is **Model Complexity (polynomial degree, 1–15)**. By moving the slider and watching the rover's path react in real time, the player *discovers* underfitting and overfitting instead of being told about them.

It is **not a quiz**. There are no questions with fixed answers; the player learns by experimenting, observing, and making a limited number of high-stakes deployment decisions.

## 2. Learning Objectives

After playing, a learner should be able to:

| # | Objective | How the game builds it |
|---|-----------|------------------------|
| LO1 | **Define underfitting (high bias)** — a model too simple to capture the true pattern. | At degree 1 the path is a straight line that misses the waypoints; both Train and Test error are high. |
| LO2 | **Define overfitting (high variance)** — a model so flexible that it memorises noise. | At degree ~12–15 the path weaves wildly through every training dot but crashes on unseen test terrain; Train error ≈ 0, Test error explodes. |
| LO3 | **Explain why Train Error alone is misleading.** | The Train Error metric keeps falling as the slider rises, while Test Error and the crash counter get worse. |
| LO4 | **Recognise the generalisation gap** (Test Error − Train Error) as the symptom of overfitting. | Live telemetry and the Diagnosis panel compare both errors. |
| LO5 | **Find the sweet spot** of complexity that minimises Test Error (the bias–variance tradeoff). | The win condition requires a Test Error close to the best achievable; the post-game plot reveals the U-shaped test-error curve. |

These objectives are also shown in the app's sidebar.

## 3. How to Play

### Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app opens in your browser (default `http://localhost:8501`).

### Playing the mission
1. **Read the Mission Briefing** at the top of the page.
2. **Drag the "Model Complexity" slider** (Degree 1–15). The purple rover path, the Train/Test errors, the crash counter and the diagnosis update instantly. *Exploring the slider is free.*
3. **Observe the plot:**
   - 🔵 Blue dots = **training waypoints** (what the model learns from)
   - 🟩 Green squares = **test terrain, safe**; ❌ red X = **test terrain, crash**
   - 🟣 Purple line = **predicted rover path**
   - Grey band = the real terrain (hidden from the model, shown for the learner)
4. **Read the Live Telemetry** (Train MSE, Test MSE, RMSE, crashes) and the **Diagnosis** panel.
5. When you think you've found the best complexity, press **🚀 Deploy Rover**. You have **6 deployments**.
6. **Win** when the Test Error is at or below the target shown in the briefing. **Lose** if you run out of deployments.
7. After the game ends, a **"Lessons Learned"** chart reveals the full Train/Test error curve across all degrees.

### Controls in the sidebar
- **Terrain difficulty** (Easy / Normal / Hard): changes sensor noise and the number of training points.
- **Generate new terrain:** new random dataset.
- **Restart mission:** same terrain, fresh deployments.

### Suggested exploration strategy (for learners)
- Start at **Degree 1**. What do you see? Are the errors low?
- Slowly move to **Degree 15**. Watch what happens *between* the blue dots.
- Compare the Train Error to the Test Error at each stop. Which one keeps improving? Which one doesn't?
- Settle on the middle and deploy.

## 4. Game Design and Learning Loop

```
 Slider (change complexity) ──► Model re-fits ──► Path + metrics update
        ▲                                                  │
        │                                                  ▼
  Try a new degree  ◄── Diagnosis / Crash feedback ◄── Observe (plot, Train vs Test)
        │
        └──► Deploy (limited tries) ──► Success / Failure ──► Reveal U-curve (reflect)
```

**Design principles**
- **Discovery learning:** the concept is never stated before the player has seen it happen.
- **Immediate, visual feedback:** the curve changes as the slider moves.
- **Meaningful stakes:** limited deployments make the player think before committing.
- **Delayed theory:** the formal explanation and the U-shaped curve appear only *after* the game ends.
- **The goal is shown, the answer is not:** the target Test Error is displayed, but the winning degree is never given. It depends on the random terrain.

| Situation | What the player sees | Concept |
|-----------|----------------------|---------|
| Degree 1 | Rigid straight line missing waypoints; high Train & Test error | **Underfitting / high bias** |
| Degree ≈ 3–6 | Smooth path following the terrain; low, similar Train & Test error | **Good fit / sweet spot** |
| Degree 12–15 | Path hits every blue dot, oscillates wildly elsewhere; Train ≈ 0, Test huge, many crashes | **Overfitting / high variance** |

## 5. Implementation Details

| Component | Details |
|-----------|---------|
| UI | **Streamlit** (sidebar, slider, metrics, expanders, session state for the deployment log) |
| Synthetic data | Hidden "true terrain" `y = 1.5·sin(x) + 0.3·x` on x ∈ [0, 6], plus Gaussian sensor noise. Separate **training** and **test** sets are sampled independently (seeded and reproducible). |
| Model | `sklearn.preprocessing.PolynomialFeatures(degree)` + `sklearn.linear_model.LinearRegression`, re-fitted on every slider change. Inputs are rescaled to [−1, 1] so a degree-15 fit stays numerically stable. |
| Metrics | Train MSE, Test MSE (`sklearn.metrics.mean_squared_error`), RMSE, and a crash counter (test points where `|prediction − truth| > 2 × noise`). |
| Plotting | `matplotlib`: training points, test points, predicted trajectory, and the true terrain, redrawn in real time. A second plot shows the player's own error log and, post-game, the full bias–variance curve. |
| Win condition | `Test MSE ≤ 1.20 × (lowest Test MSE over degrees 1–15)`. Computed from real fits, so it adapts to every random terrain (no hard-coded winning degree). |
| Diagnosis logic | Underfit if Train MSE is well above the noise floor; overfit if Test MSE > 2 × Train MSE; good if the win threshold is met. |

**Files**
- `app.py` — the game
- `requirements.txt` — dependencies
- `DOCUMENTATION.md` — this file

## 6. Concept Correctness Notes

- **Bias** is the error from overly simple assumptions (the model family can't represent the true function). **Variance** is the error from sensitivity to the particular noisy training sample. Polynomial degree is a classic complexity control that moves between the two.
- Train error generally **decreases monotonically** with degree (a nested model class can always fit at least as well). Test error is **U-shaped**.
- Test error is the estimate of generalisation. In real projects a separate validation set or cross-validation is used for choosing complexity, and the final test set is kept untouched. The game uses a single test set for simplicity and caps deployments to limit "tuning to the test set".
- Even the best model has an irreducible error equal to the noise variance, so no model reaches zero Test Error.

## 7. Playtest Evidence



### 7.1 Participant Overview

| | Student A | Student B | Student C |
|--|-----------|-----------|-----------|
| Name / initials | **[Siddhant]** | **[Lokesh]** | **[Nishant]** |
| Grade  | ECE Btech | ECE Btech | ME Btech |
| Prior knowledge | Plots points on a graph; knows straight lines | Knows linear and quadratic functions | Knows polynomials and basic curve sketching |
| Time spent playing (min) | 12 | 10 | 15 |
| Deployments used (of 6) | 3 | 2 | 4 |
| Won the mission? | Yes | Yes | Yes |


### 7.2 Observations per Student

| Aspect | Student A  | Student B  | Student C  |
|--------|-----------|-----------|-----------|
| First reaction to the slider | Excited; compared the curve to a stretchy rubber band. | Tried a few values quickly to see what each slider position did. | Predicted before moving it that "more degree means a better fit". |
| How they adjusted the curve | Dragged slowly from 1 upward, pausing at each change. | Jumped between low and high values to compare. | Moved one step at a time and compared the Train and Test numbers. |
| Moment of discovery ("aha") | Around degree 12–13, when the line began to loop between the blue dots: "It's going crazy between the dots!" | At degree 15, when many red crashes appeared although the line touched every dot. | Realised that Train Error kept falling while Test Error rose. |
| Did watching the curve change help them see overfitting? | Yes. Seeing the wiggles appear gradually made "too complex" visible without any definition. | Yes. The red crash markers made the cost of a wiggly path concrete. | Yes. Linking the wiggles to the two error numbers made it precise. |
| Their own definition of overfitting (verbatim) | "The rover remembers the dots so much that it gets confused on new ground." | "It copies the practice dots perfectly but fails the real drive." | "The model fits the noise in the training data, so it doesn't generalise." |
| Picked the sweet spot? | Yes, degree 4 on the 3rd deployment. | Yes, degree 5 on the 2nd deployment. | Yes, degree 4 on the 4th deployment (after first trying 8). |
| Confusion or difficulties | Did not know what "MSE" or "degree" meant at first. | Confused about why a perfect line through the dots was bad. | Wanted to know why Test Error was never zero. |
| Suggestions | Add simpler words and a rover animation. | Show the winning range as a hint after a failed deployment. | Add a noise slider and allow a free test of the rover. |

### 7.3 Understanding Rating (tutor's assessment, 1 = none, 5 = strong)

| Concept | Student A | Student B | Student C |
|---------|-----------|-----------|-----------|
| "A too-simple curve misses the pattern" (underfitting) | 5/5 | 5/5 | 5/5 |
| "A too-wiggly curve memorises the dots" (overfitting) | 4/5 | 4/5 | 5/5 |
| "Good on old dots ≠ good on new dots" (train vs test) | 3/5 | 4/5 | 5/5 |
| "There's a middle/best setting" (tradeoff) | 4/5 | 4/5 | 5/5 |


## 8. Credits

Developed by Sanat (EC24B1069), Rahul Ambawat (EC24B1088) and Vishwajeet Kumar (EC24I2019) for CS3006 Introduction to Data Science for Engineers, IIITDM Kancheepuram, under Dr. Vijaya Kumar S.
