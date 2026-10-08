# Diabetes & Heart Disease Chapter Four Web Application

This is a Streamlit web application implementing the Chapter Four analysis workflow from the uploaded study:

- Data upload
- Data cleaning and quality checks
- Descriptive statistics
- Class distribution
- BMI and age distributions
- Correlation analysis
- 80:20 stratified train/test split
- StandardScaler
- SMOTE on training data
- Five supervised classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbour
  - Support Vector Machine
- Five-fold stratified cross-validation
- GridSearchCV hyperparameter tuning
- Accuracy, precision, recall, F1-score, AUC-ROC
- ROC curves
- Confusion matrices
- Classification reports
- Random Forest feature importance
- Interactive prediction
- Nigerian study comparison
- CSV/ZIP export of generated tables

## Dataset files

The app expects two CSV files.

### Diabetes

The Pima dataset should contain:

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

### Heart disease

The Cleveland dataset can contain either the study's names:

`Age, Sex, ChestPain, RestBP, Chol, FBS, RestECG, MaxHR, ExAng, Oldpeak, Slope, Ca, Thal, Target`

or common UCI names such as:

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num`

The app converts heart-disease target values greater than 0 to class 1.

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

4. Open the URL shown by Streamlit, normally:

`http://localhost:8501`

## Important research note

The application calculates results from the datasets you upload. Therefore, computed values may differ from the fixed numbers currently typed into the dissertation if a different dataset version, random seed, preprocessing sequence, or hyperparameter selection was originally used.

For clinical use, this application requires appropriate local validation, ethical approval, prospective testing, and professional oversight. It is a research/educational prototype, not a diagnostic medical device.
