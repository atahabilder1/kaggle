# Machine Learning Practice

A collection of Jupyter notebooks covering fundamental machine learning concepts including EDA, classification, and regression.

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

```
ml/
├── 01-tahabilder-titanic-eda-datacleaning-visualize.ipynb  # EDA & data cleaning on Titanic dataset
├── 02-tahabilder-iris-classification.ipynb                  # Classification with Iris dataset
├── 03-tahabilder-linear-regression-from-scratch.ipynb       # Linear regression implementation
├── 04-tahabilder-titanic-survival-prediction.ipynb          # Titanic survival classification
├── 05-tahabilder-california-housing-regression.ipynb        # Housing price regression
├── housing.csv                                              # California housing dataset
├── Student_Performance.csv                                  # Student performance dataset
├── requirements.txt                                         # Python dependencies
└── README.md
```

## Notebooks Overview

| Notebook | Topic | Techniques |
|----------|-------|------------|
| 01 | Titanic EDA | Data cleaning, visualization, feature engineering |
| 02 | Iris Classification | KNN, SVM, Decision Trees, Random Forest |
| 03 | Linear Regression | From-scratch implementation, gradient descent |
| 04 | Titanic Prediction | Logistic Regression, ensemble methods |
| 05 | Housing Regression | Ridge, Lasso, Random Forest, Gradient Boosting |

## Libraries Used

- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib/seaborn** - Visualization
- **scikit-learn** - ML algorithms
- **scipy** - Statistical functions
