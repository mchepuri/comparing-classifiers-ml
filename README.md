# 🧠 Practical Application III: Comparing Classifiers — Bank Marketing Campaign Analysis

##  Overview
This project applies the **CRISP-DM (Cross Industry Standard Process for Data Mining)** framework to analyze and model the **Bank Marketing dataset** from the UCI Machine Learning Repository.  

The **business objective** is to **improve the efficiency and effectiveness** of the bank’s telemarketing campaigns by identifying customers who are most likely to subscribe to a term deposit.  

We compare and evaluate multiple machine learning algorithms — **Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Support Vector Machines (SVM)** — to determine the best performing model.


## Exploratory Data Analysis (EDA)

**Key Findings:**
- Target variable (`y`) is highly imbalanced — only ~11% of clients subscribed.  
- Features such as **job**, **education**, and **housing loan** have strong influence on subscription likelihood.  

###  Target Distribution
![Target Variable Distribution](charts/target_distribution.png)

### Subscription Rate by Job Type
![Subscription by Job Type](charts/subscription_by_job.png)

### Education vs Subscription Rate
![Education Impact](charts/education_subscription.png)

### Correlation Matrix
![Correlation Matrix](charts/correlation_matrix.png)

---

##  Baseline Model

Before using any algorithm, we established a simple **baseline model** that always predicts the **majority class (“no”)**.

| Metric | Value |
|---------|--------|
| **Majority Class** | No |
| **Baseline Accuracy** | **≈ 88.7%** |

This serves as the minimum accuracy our models should outperform.

---

## Models Implemented

Four classifiers were trained and evaluated using default hyperparameters:

| Model | Train Time (s) | Train Accuracy | Test Accuracy |
|--------|----------------|----------------|----------------|
| **Logistic Regression** | ~0.2 | 0.89 | **0.90** |
| **K-Nearest Neighbors (KNN)** | ~0.8 | 0.94 | 0.87 |
| **Decision Tree** | ~0.05 | **1.00** | 0.88 |
| **Support Vector Machine (SVM)** | ~2.1 | 0.91 | 0.89 |

**Insights:**
- Logistic Regression delivered the **most balanced** performance.  
- KNN and Decision Tree slightly overfit the training data.  
- SVM achieved good performance but required longer training time.

---

## 🔍 Model Optimization (GridSearchCV + Feature Selection)

A **pipeline** was built combining preprocessing, feature selection, and Logistic Regression, followed by **GridSearchCV** for hyperparameter tuning.

```python
pipe = Pipeline([
    ('preprocess', preprocessor),
    ('select', SelectKBest(score_func=f_classif)),
    ('model', LogisticRegression(max_iter=1000))
])

param_grid = {
    'select__k': [10, 15, 20, 'all'],
    'model__C': [0.01, 0.1, 1, 10],
    'model__penalty': ['l2'],
    'model__solver': ['lbfgs']
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
