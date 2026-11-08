**Course:** CS3006 Introduction to Data Science for Engineers
**Professor:** Dr. Vijaya Kumar S
**Institute:** IIITDM Kancheepuram
**Authors:** Sanat (EC24B1069) and Rahul Ambawat (EC24B1088)

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

> **Instructions for the tutor:** play the game with each of the three students (Grade 6–10 maths). Use the prompts below, record what they said and did *in their own words* where possible, and fill in the tables. Delete the italic hints once completed. Attach screenshots/photos (with consent) in `/evidence/`.

### 7.1 Session Details

| Field | Entry |
|-------|-------|
| Date(s) of testing | *DD/MM/YYYY* |
| Format | *Online (screen share) / in person* |
| Tester / facilitator | *Name* |
| Game version / difficulty | *e.g., Normal* |
| Consent obtained from parent/guardian? | *Yes / No* |

### 7.2 Participant Overview

| | Student A | Student B | Student C |
|--|-----------|-----------|-----------|
| Name / initials | | | |
| Grade | | | |
| Prior knowledge (functions? graphs? polynomials?) | | | |
| Time spent playing (min) | | | |
| Number of deployments used | | | |
| Won the mission? | | | |

### 7.3 Facilitator Prompts (ask in this order, without giving away answers)

1. *Before playing:* "What do you think happens if a curve is made very wiggly so it touches every dot?"
2. *At Degree 1:* "What do you notice about the line and the dots? Is it a good path for the rover?"
3. *While dragging towards Degree 15:* "What is happening to the curve between the dots? What about the red crosses?"
4. *Compare errors:* "Which number keeps getting smaller? Which number goes up? Why might that be?"
5. *After playing:* "In your own words, what is 'overfitting'? Can you give a real-life example?" (e.g., memorising answers vs. understanding the method)

### 7.4 Observations per Student

#### Student A

| Aspect | Notes |
|--------|-------|
| First reaction to the slider | *e.g., "It's like stretching a rubber band."* |
| What they did when adjusting the curve | *e.g., dragged slowly vs. jumped; went back and forth?* |
| Moment of discovery ("aha") | *Which degree? What did they say?* |
| Did watching the curve change help them visualise overfitting? | *Yes / Partly / No — describe how* |
| Their own definition of overfitting (verbatim) | *"..."* |
| Did they correctly pick the sweet spot? | *Yes / No, degree chosen* |
| Confusion or difficulties | |
| Suggestions | |

#### Student B

| Aspect | Notes |
|--------|-------|
| First reaction to the slider | |
| What they did when adjusting the curve | |
| Moment of discovery ("aha") | |
| Did watching the curve change help them visualise overfitting? | |
| Their own definition of overfitting (verbatim) | |
| Did they correctly pick the sweet spot? | |
| Confusion or difficulties | |
| Suggestions | |

#### Student C

| Aspect | Notes |
|--------|-------|
| First reaction to the slider | |
| What they did when adjusting the curve | |
| Moment of discovery ("aha") | |
| Did watching the curve change help them visualise overfitting? | |
| Their own definition of overfitting (verbatim) | |
| Did they correctly pick the sweet spot? | |
| Confusion or difficulties | |
| Suggestions | |

### 7.5 Understanding Rating (tutor's assessment, 1 = none, 5 = strong)

| Concept | Student A | Student B | Student C |
|---------|-----------|-----------|-----------|
| "A too-simple curve misses the pattern" (underfitting) | | | |
| "A too-wiggly curve memorises the dots" (overfitting) | | | |
| "Good on old dots ≠ good on new dots" (train vs test) | | | |
| "There's a middle/best setting" (tradeoff) | | | |

### 7.6 Summary of Findings

| Question | Summary |
|----------|---------|
| How did actively dragging the slider help the students *see* overfitting? | |
| Common misconceptions observed | |
| Which game element helped most (path, red crashes, error numbers, deploy limit)? | |
| What was confusing for younger learners (e.g., "MSE", "degree")? | |
| Changes we made or would make as a result | |

### 7.7 Learning Evidence Checklist

- [ ] All three students completed at least one full mission
- [ ] Each student gave a verbal definition of overfitting in their own words
- [ ] Screenshots / photos attached (with consent)
- [ ] Quotes recorded verbatim
- [ ] Improvements listed in 7.6

## 8. Limitations and Future Work

- A single train/test split is used. Cross-validation would be more robust.
- Only polynomial regression is covered; extensions could add regularisation (Ridge/Lasso) sliders, more training data, or a noise slider.
- A "kid-mode" with the terms *Train Error* and *Test Error* renamed "Practice mistakes" and "Real-drive mistakes" may suit younger learners.

## 9. Credits

Developed by Sanat (EC24B1069) and Rahul Ambawat (EC24B1088) for CS3006 Introduction to Data Science for Engineers, IIITDM Kancheepuram, under Dr. Vijaya Kumar S.
