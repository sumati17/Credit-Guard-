
# CreditGuard — Credit Risk Intelligence

CreditGuard is an end-to-end machine learning portfolio project that explores customer delinquency risk prediction and responsible support interventions.

The project takes a structured customer dataset through exploratory data analysis, preprocessing, feature engineering, model training, evaluation, and an interactive Streamlit dashboard.

## Project Objectives

- Explore customer financial and payment behaviour.
- Identify data quality issues and potential risk indicators.
- Build machine learning models to estimate delinquency probability.
- Evaluate models using multiple classification metrics.
- Explore model explainability and group-level fairness checks.
- Demonstrate risk-based customer support recommendations.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook
- Git and GitHub

## Project Structure

```text
CreditGuard/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Training.ipynb
│   ├── 05_Model_Evaluation.ipynb
│   └── 06_Risk_Intelligence.ipynb
│
├── models/
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── reports/
│
├── src/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Project Workflow

### 1. Exploratory Data Analysis
- Dataset structure and quality checks.
- Missing value analysis.
- Univariate and bivariate analysis.
- Payment behaviour exploration.
- Delinquency distribution analysis.

### 2. Data Preprocessing
- Standardisation of categorical labels.
- Payment history encoding.
- Missing value treatment planning.
- Data validation and quality checks.

### 3. Feature Engineering
- Payment behaviour counts.
- Payment issue rates.
- Missed payment rates.
- On-time payment rates.
- Train-test splitting.

### 4. Model Training
- Dummy Classifier baseline.
- Logistic Regression.
- Random Forest.

### 5. Model Evaluation
- Precision, recall and F1-score.
- ROC-AUC and PR-AUC.
- Confusion matrices.
- Calibration and Brier score.
- Permutation importance.
- Exploratory fairness analysis.

### 6. Risk Intelligence
- Predicted delinquency probabilities.
- Provisional risk categories.
- Responsible customer support recommendations.

### 7. Interactive Dashboard
A Streamlit application that accepts customer financial information and payment history and displays a predicted risk probability and suggested support intervention.

## Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CreditGuard
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the dashboard

```bash
python -m streamlit run app/app.py
```

## Models

The project compares a Logistic Regression model and a Random Forest model against a Dummy Classifier baseline.

The trained Logistic Regression pipeline is used by the current Streamlit dashboard.

## Limitations and Responsible Use

- This is an educational portfolio prototype, not a production credit-scoring system.
- The dataset is relatively small and has an imbalanced target.
- Prediction timing and the target horizon require further validation to rule out data leakage.
- Some dataset definitions and units require confirmation.
- Risk thresholds are provisional and should not be treated as validated business rules.
- Fairness checks are exploratory and do not establish that a model is fair.
- Predictions must not be used to make automatic lending, rejection, or collections decisions.
- Customer privacy and human review are essential.

##  Author

Built as a machine learning portfolio project to demonstrate practical skills in data analysis, machine learning, model evaluation, and application development.
