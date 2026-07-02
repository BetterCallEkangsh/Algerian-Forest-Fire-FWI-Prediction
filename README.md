<div align="center">

# 🔥 Algerian Forest Fire — FWI Prediction

### Regression modeling of the Fire Weather Index using regularized linear models

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📌 Overview

This project analyzes the **Algerian Forest Fires dataset** — 244 weather observations from two regions of Algeria (Bejaia in the northeast, Sidi Bel-Abbes in the northwest), collected over June–September 2012 — to predict the **Fire Weather Index (FWI)**, a composite score used by fire-danger rating systems worldwide.

The work is split into two notebooks:

| Notebook | Purpose |
|---|---|
| `Algerian_Forest_Fire_Data_Analysis.ipynb` | Cleaning, wrangling, and exploratory data analysis |
| `Algerian_Forest_Model_Training.ipynb` | Feature selection, scaling, and regression modeling |

---

## 🌍 Dataset

- **244 instances** — 122 per region, merged from two raw CSV blocks
- **11 weather/FWI-system features** + 1 target class label
- Region split encoded as a binary feature (`0` = Bejaia, `1` = Sidi Bel-Abbes)
- Class balance: **138 fire** vs **106 not-fire** days

| Feature | Description | Range |
|---|---|---|
| Temperature | Noon temperature (°C) | 22 – 42 |
| RH | Relative Humidity (%) | 21 – 90 |
| Ws | Wind speed (km/h) | 6 – 29 |
| Rain | Total rainfall (mm) | 0 – 16.8 |
| FFMC | Fine Fuel Moisture Code | 28.6 – 92.5 |
| DMC | Duff Moisture Code | 1.1 – 65.9 |
| DC | Drought Code | 7 – 220.4 |
| ISI | Initial Spread Index | 0 – 18.5 |
| BUI | Buildup Index | 1.1 – 68 |
| **FWI** | **Fire Weather Index (target)** | 0 – 31.1 |
| Classes | `fire` / `not fire` | — |

---

## 🧹 Data Cleaning

- Merged the two region-specific blocks of the raw CSV and stamped a `Region` column
- Dropped a malformed header row separating the two regions, and rows with nulls
- Stripped whitespace from column names and category labels (`" fire"` → `"fire"`)
- Cast day/month/year and integer-valued weather fields to `int`, remaining FWI-system columns to `float`
- Persisted a clean, analysis-ready CSV for the modeling notebook

## 📊 Exploratory Analysis

- **KDE plots** across all numeric features to inspect distribution shape
- **Class balance pie chart** — ~56.5% fire vs 43.5% not-fire
- **Correlation heatmap** to surface multicollinearity among FWI-system indices
- **Boxplots** per feature to flag outliers
- **Region-wise monthly fire counts** (Bejaia vs Sidi Bel-Abbes) to compare seasonal fire risk between the two geographies

---

## 🤖 Modeling Pipeline

```
Raw CSV → Clean & Encode → Drop target-adjacent leakage (Classes/FWI split)
        → Train/Test Split (80/20) → Correlation-based Feature Selection (|r| > 0.85)
        → StandardScaler → Linear / Lasso / Ridge / ElasticNet (+ CV variants)
```

**Feature selection:** highly collinear features (correlation threshold 0.85) were dropped from both train and test sets *after* the split, to avoid leakage.

**Scaling:** `StandardScaler` fit on training data only, then applied to test data.

**Models compared**, each evaluated with MAE, MSE, and R²:

| Model | Description |
|---|---|
| Linear Regression | Baseline, no regularization |
| Lasso | L1-regularized, default α |
| **LassoCV** | L1-regularized, α tuned via 5-fold CV over a 50-point log-spaced grid |
| Ridge | L2-regularized, default α |
| **RidgeCV** | L2-regularized, α tuned via 5-fold CV |
| ElasticNet | L1 + L2 mix, default params |
| **ElasticNetCV** | L1 + L2 mix, α tuned via 5-fold CV |

---

## 📈 Results

| Model | MAE ↓ | MSE ↓ | R² ↑ |
|---|:---:|:---:|:---:|
| Linear Regression | 0.505 | — | **0.9806** |
| Lasso (default) | 1.009 | — | 0.9431 |
| **LassoCV** | 0.547 | 0.642 | 0.9788 |
| Ridge (default) | 0.519 | — | 0.9801 |
| **RidgeCV** | 0.590 | 0.684 | 0.9774 |
| ElasticNet (default) | 1.683 | — | 0.8677 |
| **ElasticNetCV** | 0.573 | 0.666 | 0.9780 |

> Cross-validated variants dramatically outperform their default-hyperparameter counterparts, especially ElasticNet — showing how much untuned regularization strength can hurt a model when the default α doesn't suit the data.

**Tuned hyperparameters:**

| Model | Best α (chosen by CV) |
|---|:---:|
| LassoCV | 0.0391 |
| RidgeCV | 4.7149 |
| ElasticNetCV | 0.0295 |

All three CV models land in the **97–98% R² range** on held-out test data — meaning the FWI can be predicted with high accuracy directly from the underlying weather and fuel-moisture measurements, once regularization strength is tuned properly.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Jupyter`

---

## 🚀 Getting Started

```bash
git clone https://github.com/BetterCallEkangsh/Algerian-Forest-Fire-FWI-Prediction.git
cd Algerian-Forest-Fire-FWI-Prediction
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order:
1. `Algerian_Forest_Fire_Data_Analysis.ipynb` — generates the cleaned dataset
2. `Algerian_Forest_Model_Training.ipynb` — trains and evaluates the models

---

## 📂 Repository Structure

```
├── Datasets/
│   ├── Algerian_forest_fires_dataset_UPDATE.csv
│   └── Cleaned Datasets/
│       └── Cleaned Algerian Forest Dataset.csv
├── Algerian_Forest_Fire_Data_Analysis.ipynb
├── Algerian_Forest_Model_Training.ipynb
└── README.md
```

---

<div align="center">

*Built as part of a hands-on machine learning practice series covering regression, regularization, and cross-validation.*

</div>
