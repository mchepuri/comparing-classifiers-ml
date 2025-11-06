README — Practical Application III: Comparing Classifiers (Bank Marketing Dataset)
Overview

This notebook applies the CRISP-DM (Cross Industry Standard Process for Data Mining) framework to evaluate multiple classification algorithms for predicting term deposit subscriptions using the Bank Marketing dataset from the UCI Machine Learning Repository.

The project’s primary business objective is to improve the efficiency and effectiveness of the bank’s telemarketing campaigns by identifying customers who are most likely to subscribe to a term deposit.
xploratory Data Analysis Highlights

Key Charts Included:

Target Distribution — majority of customers did not subscribe (y='no' ≈ 88%).

sns.countplot(x='y', data=df)
plt.title('Target Variable Distribution (Term Deposit Subscription)')


Subscription Rate by Job Type — jobs in management, admin, and students showed higher subscription probabilities.

Education vs. Subscription Rate — higher education levels correlate with higher likelihood to subscribe.

Correlation Matrix (Numeric Features) — shows no extreme multicollinearity among numeric variables.

Baseline Model

Before training any models, a baseline predictor that always predicts the majority class ("no") was established.

Baseline Accuracy: ≈ 88.7%

This sets the benchmark every classifier must outperform.

Models Implemented and Compared
Model	Train Time (s)	Train Accuracy	Test Accuracy
Logistic Regression	~0.2	0.89	0.90
K-Nearest Neighbors	~0.8	0.94	0.87
Decision Tree	~0.05	1.00	0.88
Support Vector Machine	~2.1	0.91	0.89

Interpretation:

Logistic Regression achieved the most balanced performance between training and testing accuracy.

Decision Tree and KNN showed mild overfitting.

SVM performed well but had higher computation time.

🔍 Model Optimization

A GridSearchCV pipeline was built to tune Logistic Regression hyperparameters, including:

Regularization strength (C)

Feature selection (SelectKBest)

Solver and penalty types

Best Parameters Found:

{'select__k': 15, 'model__C': 1, 'model__penalty': 'l2', 'model__solver': 'lbfgs'}


Performance After Tuning:

Cross-Validation Accuracy: ~0.91

Test Accuracy: ~0.91

This demonstrates a modest improvement while maintaining generalization.

Feature Engineering

Combine social and economic indicators (e.g., euribor3m trends, employment rates) for richer feature sets.

Create interaction features (e.g., job × education).

Modeling Enhancements

Apply ensemble methods (Random Forests, Gradient Boosting, XGBoost) for potentially higher predictive power.

Consider SMOTE or other resampling to address class imbalance.