# IDSE CS3006 Group Project: Terrain-Risk Classification & Educational Game

**Course:** CS3006 Introduction to Data Science for Engineers  
**Professor:** Dr. Vijaya Kumar S  
**Institute:** IIITDM Kancheepuram  
**Authors:** 
* Sanat (EC24B1069) 
* Rahul Ambawat (EC24B1088)

---

## 📌 Project Overview

This repository contains our final submission for the CS3006 IDSE course. The project is divided into two distinct parts, satisfying the complete grading rubric:

1. **Part A (Data Science Project):** An end-to-end, zero-leakage machine learning pipeline that classifies autonomous rover IMU time-series data into navigation-risk tiers to prevent stalling.
2. **Part B (Educational Game):** An interactive web application built with Streamlit, designed to teach 6th-10th grade students the concept of the Bias-Variance Tradeoff ("Overfitting vs. Underfitting").

---

## 📂 Repository Structure

```text
IDSE_CS3006_Project/
│
├── docs/                                 # Final compiled PDFs
│   ├── Rover_Project_Report.pdf          # Detailed academic report (Part A)
│   └── Rover_Presentation.pdf            # Slide deck for project viva
│
├── Part_A_DataScience/                   # The ML Pipeline
│   ├── data/                             # Dataset (auto-downloads from Kaggle mirror)
│   ├── outputs/                          # Generated figures, CSV metrics, and logs
│   ├── run_pipeline.py                   # Main orchestration script
│   ├── config.py, data.py, eda.py        # Config, data loading/cleaning, and EDA
│   ├── features.py, models.py            # Feature engineering and model architectures
│   ├── evaluate.py                       # Risk-aware metrics and error analysis
│   └── requirements-tested.txt           # Exact versions used for reproducibility
│
└── Part_B_Game/                          # The Educational Game
    ├── app.py                            # Streamlit application code
    ├── DOCUMENTATION.md                  # Game docs, learning objectives, and playtest evidence
    ├── evidence/                         # Screenshots of students playtesting the game
    └── requirements.txt                  # Dependencies for the web app
```

---

## 🚀 Part A: Autonomous Rover Terrain Classification

We evaluate a Logistic Regression baseline, a Random Forest, and a PyTorch 1D-CNN on 128-sample IMU windows. The protocol uses **Stratified Group K-Fold Cross-Validation** by recording session to prevent the massive data leakage that occurs with random window shuffling. 

**To run the pipeline locally:**
```bash
cd Part_A_DataScience
pip install -r requirements.txt
python run_pipeline.py
```
*(Note: Use `python run_pipeline.py --quick` for a 2-minute smoke test. Full run takes ~15 minutes on a standard CPU).*

For a deep dive into the physical feature engineering (gravity projection, spike isolation), model selection, and the critical error analysis showing why IMU sensors struggle with smooth wood floors, please see the **Project Report PDF** in the `docs/` folder.

---

## 🎮 Part B: "The Overfitting Rover" (Bias-Variance Game)

This interactive app places the user in the shoes of a rover systems engineer. By dragging a "Model Complexity" slider (Polynomial Degree), players actively discover why a model that perfectly memorizes the training data (overfitting) crashes on real-world terrain.

**To run the game locally:**
```bash
cd Part_B_Game
pip install -r requirements.txt
python -m streamlit run app.py
```
*(If your browser does not open automatically, navigate to `http://localhost:8501`)*

We conducted playtests with grade 6-10 students to validate the educational efficacy of the app. The full playtest logs, student quotes, and pedagogical design principles are available in [`Part_B_Game/DOCUMENTATION.md`](Part_B_Game/DOCUMENTATION.md).

---

## 📜 Credits

Developed in November 2026 for the IDSE curriculum at IIITDM Kancheepuram. 