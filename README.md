# ❤️ Heart Disease Classification

A complete exploratory and predictive machine learning project to detect the presence of heart disease using medical features such as chest pain type, cholesterol level, blood pressure, and more.

> 📊 Performed EDA + 🧠 Trained & Tuned Multiple ML Models + 🔍 Compared Accuracy & Metrics

---

## 🗂️ Dataset Overview

- **Source**: [heart-disease.csv](heart-disease.csv)
- **Goal**: Predict if a patient has heart disease (1) or not (0)
- **Target Column**: `target`

### 📋 Feature Descriptions

| Feature     | Description |
|-------------|-------------|
| age         | Age of the patient |
| sex         | 1 = Male, 0 = Female |
| cp          | Chest pain type (0–3) |
| trestbps    | Resting blood pressure |
| chol        | Serum cholesterol (mg/dl) |
| fbs         | Fasting blood sugar > 120 mg/dl |
| restecg     | Resting electrocardiographic results |
| thalach     | Max heart rate achieved |
| exang       | Exercise induced angina |
| oldpeak     | ST depression induced by exercise |
| slope       | Slope of the peak exercise ST segment |
| ca          | Major vessels colored by flourosopy |
| thal        | Thalassemia result |
| target      | 1 = Heart disease, 0 = No heart disease |

---

## 📊 Exploratory Data Analysis (EDA)

- Checked for null/missing values
- Visualized correlations using heatmap
- Plotted distributions:
  - ✅ Heart Disease by Gender
  - ✅ Age vs Max Heart Rate
  - ✅ Chest Pain Type vs Heart Disease

---

## ⚙️ Machine Learning Models Used

| Model                    | Accuracy (%) |
|-------------------------|--------------|
| Logistic Regression      | 85.25%       |
| Linear SVC               | 81.96%       |
| KNeighbors Classifier    | 60.66%       |
| Random Forest Classifier | 85.24%       |
| Bagging Classifier       | 70.49%       |
| Perceptron               | 50.82%       |
| Linear Regression        | 47.66%       |

> 📈 Plotted all model scores with bar graphs for easy comparison.

---

## 🛠️ Hyperparameter Tuning

Used **GridSearchCV** and **RandomizedSearchCV** to optimize:

- ✅ Logistic Regression  
- ✅ Linear SVC  
- ✅ Random Forest Classifier  

Resulted in **improved model accuracy and stability**.

---

## ▶️ How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Denistanb/Heart-Disease-Classification.git
   cd Heart-Disease-Classification
2. **Install Dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
3. **Launch the Notebook**:
   ```bash
   jupyter notebook "Heart Disease Classification.ipynb"
