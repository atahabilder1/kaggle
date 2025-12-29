# Machine Learning Practice

A collection of Jupyter notebooks covering fundamental machine learning concepts including EDA, classification, regression, deep learning for computer vision, natural language processing, and business analytics.

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
├── 06-tahabilder-cats-vs-dogs-cnn-image-classification.ipynb # CNN image classification
├── 07-tahabilder-sentiment-analysis-twitter.ipynb           # NLP sentiment analysis on tweets
├── 08-tahabilder-customer-churn-prediction.ipynb            # Customer churn prediction
├── housing.csv                                              # California housing dataset
├── Student_Performance.csv                                  # Student performance dataset
├── WA_Fn-UseC_-Telco-Customer-Churn.csv                     # Telco customer churn dataset
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
| 06 | Cats vs Dogs CNN | CNNs, Transfer Learning (VGG16), Data Augmentation |
| 07 | Sentiment Analysis | NLP, Text Preprocessing, TF-IDF, LSTM, Naive Bayes |
| 08 | Customer Churn | SMOTE, Class Weights, XGBoost, Business Metrics |

## Libraries Used

- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib/seaborn** - Visualization
- **scikit-learn** - ML algorithms
- **scipy** - Statistical functions
- **tensorflow/keras** - Deep learning and CNNs
- **opencv/pillow** - Image processing
- **nltk** - Natural language processing
- **wordcloud** - Text visualization
- **imbalanced-learn** - Handling imbalanced datasets (SMOTE)
- **xgboost** - Gradient boosting algorithms
