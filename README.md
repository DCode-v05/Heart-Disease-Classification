# Heart Disease Classification

**Predict whether a patient has heart disease from routine clinical measurements, then compare seven scikit-learn models to see which one calls it best.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white) ![seaborn](https://img.shields.io/badge/seaborn-4C72B0?style=flat&logo=python&logoColor=white)

## Overview

This is a single-notebook study that takes the UCI Cleveland heart-disease dataset and works it end to end: load the data, dig through it with exploratory analysis, train a handful of classifiers on baseline settings, compare them side by side, then tune the strongest few with cross-validated hyperparameter search.

The target is binary — does this patient have heart disease (`1`) or not (`0`) — based on 13 clinical features like age, sex, chest-pain type, resting blood pressure, cholesterol, max heart rate during exercise, and ST depression. The point of the project was to practice the full classification workflow on tabular medical data and get a feel for how different algorithms behave on the same split, rather than to ship a deployable diagnostic tool.

I built this as a learning exercise while working through ML fundamentals. It lives as one well-sectioned notebook (62 cells) with documented results at each tuning step, and the best models land around 85% test accuracy on the 303-patient Cleveland set.

## Key Features

- Loads and inspects the UCI Cleveland heart-disease dataset (303 patients, 13 features + target) with `info()`, `describe()`, and a null check before any modeling.
- Exploratory data analysis that actually looks at the clinical relationships:
  - Heart-disease frequency by **sex** (crosstab + bar chart).
  - **Age vs. maximum heart rate** scatter, colored by disease/no-disease, to see how the two classes separate.
  - **Chest-pain type vs. target** crosstab and bar chart, with the four chest-pain categories labeled (typical angina, atypical angina, non-anginal, asymptomatic).
  - A full **correlation matrix** rendered as a 15x15 annotated Seaborn heatmap.
- Trains **seven models** on baseline parameters over the same 80/20 train/test split: Logistic Regression, Linear SVC, KNeighbors, Random Forest, Bagging, Perceptron, and Linear Regression.
- Collects every model's accuracy into a single DataFrame and plots a labeled bar chart so the comparison is one glance.
- **Hyperparameter tuning** on the three most promising models — Logistic Regression and Random Forest via `GridSearchCV`, Linear SVC via `RandomizedSearchCV` — all with 5-fold cross-validation, and the chosen best parameters written into markdown next to each result.
- Fixed random seed (`np.random.seed(30)`) throughout so the split and the scores reproduce on re-run.
- The column reference at the top of the notebook documents what every feature means clinically (e.g. `oldpeak` = ST depression induced by exercise, `ca` = number of major vessels colored by fluoroscopy), so the EDA reads as more than just plotting columns.

## How It Works

The whole pipeline lives in `Heart Disease Classification.ipynb` and runs top to bottom.

### Data

The notebook reads `heart-disease.csv` — the standard UCI Cleveland set, 303 rows, 14 columns (13 predictors plus the `target` label). Columns include `age`, `sex`, `cp` (chest pain type), `trestbps` (resting BP), `chol` (serum cholesterol), `fbs` (fasting blood sugar flag), `restecg`, `thalach` (max heart rate), `exang` (exercise-induced angina), `oldpeak`, `slope`, `ca`, and `thal`. There are no missing values, so cleaning is minimal and the focus stays on analysis and modeling.

The repo also ships a second file, `heart.csv` (1,025 rows, same schema) — a larger augmented version of the same dataset. The notebook itself trains on the 303-row `heart-disease.csv`; `heart.csv` is included as an alternate, bigger source to experiment with.

### Exploratory analysis

Before modeling, the notebook builds intuition with three targeted views — sex, age-vs-heart-rate, and chest-pain type — each as a crosstab plus a chart, followed by a correlation heatmap across all features. This is where the clinical column descriptions pay off: the plots are labeled with real categories rather than raw integer codes.

### Modeling

Features (`X`) are everything except `target`; the label (`y`) is `target`. The data is split 80/20 with `train_test_split` under a fixed seed. Each of the seven estimators is fit on the training set and scored on the held-out test set with its default parameters first, so there's a clean baseline before any tuning.

A note worth being honest about: `LinearRegression` is included in the lineup even though it's a regressor, so its `.score()` reports R² rather than classification accuracy — it's there as a comparison point against the actual classifiers.

### Comparison

All seven baseline scores are gathered into a dictionary, turned into a one-row DataFrame, and plotted as a bar chart with the accuracy printed on top of each bar. This makes it obvious which models are worth tuning.

### Hyperparameter tuning

The three strongest candidates get a proper search with 5-fold cross-validation:

- **Logistic Regression** — `GridSearchCV` over `C`, `penalty`, `solver`, `class_weight`, and `max_iter`. The tuned model is then refit with the best parameters and re-scored.
- **Linear SVC** — `RandomizedSearchCV` over `C`, `penalty`, `loss`, `tol`, `max_iter`, and `multi_class`.
- **Random Forest** — `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `bootstrap`.

For each, the notebook records `best_params_` and refits the model so the reported tuned accuracy comes from the actual chosen configuration, with the parameter set written into markdown right below.

## Results / Highlights

On the 303-patient Cleveland set with an 80/20 split (seed 30):

- **Logistic Regression — 85.25% test accuracy** at baseline; tuning didn't beat it, so the notebook keeps the baseline. Best of the bunch.
- **Random Forest — 85.24% test accuracy** after `GridSearchCV` tuning (`n_estimators=100`, `max_depth=20`, `min_samples_split=5`, `min_samples_leaf=1`, `max_features="log2"`, `bootstrap=True`).
- **Linear SVC — 81.96% test accuracy** after `RandomizedSearchCV` tuning (`C=0.1`, `loss="hinge"`, `penalty="l1"`, `multi_class="crammer_singer"`, `max_iter=2000`, `tol=0.001`).
- All seven baseline models scored and compared in one chart; the linear and tree models clustered at the top, with KNN and Perceptron trailing.

The headline takeaway: a plain Logistic Regression was already as good as a tuned Random Forest here, which is a fair result for a small, clean tabular dataset.

## Tech Stack

- **Language:** Python 3.11
- **Environment:** Jupyter Notebook
- **Data:** pandas, NumPy
- **ML:** scikit-learn — `LogisticRegression`, `LinearSVC`, `KNeighborsClassifier`, `RandomForestClassifier`, `BaggingClassifier`, `Perceptron`, `LinearRegression`; `train_test_split`, `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`; metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `classification_report`, `confusion_matrix`, `RocCurveDisplay`)
- **Visualization:** Matplotlib, seaborn

## Getting Started

### Prerequisites

- Python 3.11 (anything 3.x should work)
- Jupyter Notebook or JupyterLab
- pip

### Installation

```bash
git clone https://github.com/DCode-v05/Heart-Disease-Classification.git
cd Heart-Disease-Classification
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running

```bash
jupyter notebook "Heart Disease Classification.ipynb"
```

The notebook loads `heart-disease.csv` by its bare filename, so launch Jupyter from the repo root (or point the `pd.read_csv` path at `Data/heart-disease.csv`) and run the cells top to bottom.

## Usage

Open the notebook and run it sequentially. It will:

1. Load the dataset and print its shape, dtypes, summary stats, and null counts.
2. Walk through the EDA charts (sex, age vs. heart rate, chest pain, correlation heatmap).
3. Train and score all seven baseline models, then show the comparison bar chart.
4. Run the cross-validated hyperparameter searches for Logistic Regression, Linear SVC, and Random Forest and print each tuned accuracy.

To experiment, swap the input to `Data/heart.csv` (the 1,025-row version), change `test_size` or the seed in the split cell, or add models / parameter grids to the comparison and tuning sections.

## Project Structure

```
Heart-Disease-Classification/
├── Data/
│   ├── heart-disease.csv              # UCI Cleveland set, 303 rows — the one the notebook trains on
│   └── heart.csv                      # Larger 1,025-row version, same 14 columns
├── Heart Disease Classification.ipynb # Full workflow: EDA -> 7-model baseline -> comparison -> tuning (62 cells)
└── README.md
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
