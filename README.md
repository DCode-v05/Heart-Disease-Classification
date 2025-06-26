# Heart Disease Classification

## Project Description
This project aims to predict the presence of heart disease in patients using various medical features. Leveraging machine learning techniques, the project provides a comprehensive workflow from data exploration to model evaluation, helping to identify individuals at risk of heart disease based on clinical data.

## Project Details

### Problem Statement
Heart disease is a leading cause of mortality worldwide. Early detection can save lives. This project uses patient data to build predictive models that classify whether a patient has heart disease.

### Dataset Overview
- **Files**: `Data/heart-disease.csv`, `Data/heart.csv`
- **Source**: Publicly available UCI Heart Disease dataset
- **Target Variable**: `target` (1 = Heart disease, 0 = No heart disease)
- **Features**: Age, sex, chest pain type, cholesterol, blood pressure, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, slope, number of vessels, thalassemia, etc.

### Exploratory Data Analysis (EDA)
- Checked for missing/null values
- Visualized feature distributions and correlations
- Explored relationships between features and the target variable

### Machine Learning Workflow
- Data preprocessing and feature engineering
- Model training and evaluation using:
  - Logistic Regression
  - Linear SVC
  - KNeighbors Classifier
  - Random Forest Classifier
  - Bagging Classifier
  - Perceptron
  - Linear Regression
- Hyperparameter tuning with GridSearchCV and RandomizedSearchCV
- Model comparison using accuracy and other metrics

### Results
- Best performing models: Logistic Regression and Random Forest Classifier (accuracy ~85%)
- Visual comparison of model performance

## Tech Stack
- Python 3.x
- Jupyter Notebook
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/TensoRag/Heart-Disease-Classification.git
   cd Heart-Disease-Classification
   ```
2. **Install Dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. **Launch the Notebook**
   ```bash
   jupyter notebook "Heart Disease Classification.ipynb"
   ```

## Usage
- Open the notebook in Jupyter.
- Run the cells sequentially to perform EDA, train models, and evaluate results.
- Modify model parameters or try different algorithms as needed.

## Project Structure
```
Heart-Disease-Classification/
├── Data/
│   ├── heart-disease.csv
│   └── heart.csv
├── Heart Disease Classification.ipynb
├── README.md
```
- `Data/`: Contains the datasets used for training and evaluation.
- `Heart Disease Classification.ipynb`: Main notebook with code, analysis, and results.
- `README.md`: Project documentation.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

## Contact
- **GitHub**: [TensoRag](https://github.com/TensoRag)
- **Email**: denistanb05@gmail.com
