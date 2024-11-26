# credit-risk-classification
## Project Overview

This project focuses on building a machine learning model to classify credit risk in peer-to-peer lending. The primary objective is to predict whether a loan is high-risk or healthy based on key financial features, providing valuable insights for lenders to assess borrower creditworthiness and make informed lending decisions.

## Project Structure

credit_risk_classification.ipynb:
Jupyter notebook containing data exploration, model training, and evaluation.
lending_data.csv:
A dataset with historical lending activity, which includes financial metrics for borrowers.
Methodology

## 1. Data Preparation
The first step involved loading the dataset from the lending_data.csv file and preparing it for analysis. Key steps included:

Feature Selection: The dataset contains several financial features, such as loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, and total_debt.
Label Separation: The target variable (loan_status) was extracted, which indicates whether a loan is "healthy" (low-risk) or "high-risk."

## 2. Data Splitting
To evaluate model performance, we split the data into:

Training Set: Used to train the machine learning model.
Testing Set: Used to assess the model's performance on unseen data using train_test_split.

## 3. Model Training
A logistic regression model was selected for this binary classification problem. Logistic regression is particularly well-suited for scenarios where the outcome is categorical, as in this case where the loan status is either "healthy" or "high-risk."

## 4. Model Evaluation
The model's performance was evaluated using the following metrics:

## Confusion Matrix: To visualize the true positives, false positives, true negatives, and false negatives.
Classification Report: To calculate precision, recall, F1-score, and accuracy for both healthy and high-risk loans.
Results

Confusion Matrix
The model performed well, correctly classifying the majority of loans:

## Healthy Loans (Low-Risk): The model correctly predicted 18,663 out of 18,765 healthy loans, indicating high performance in identifying low-risk loans.
## High-Risk Loans: The model correctly predicted 563 out of 619 high-risk loans, which shows strong predictive power but highlights areas for improvement.
Classification Report
Loan Type	Precision	Recall	F1-Score	Support
Healthy Loan	1.00	0.99	1.00	18,765
High-Risk Loan	0.85	0.91	0.88	619
Overall	99%			
## Healthy Loans: The model achieves near-perfect precision (1.00) and high recall (0.99), indicating excellent identification of low-risk loans.
## High-Risk Loans: Precision is 85%, while recall is 91%. The model is effective at identifying high-risk loans, but there is room for improvement in minimizing false positives.
Overall Performance Metrics
## Accuracy: 99% — The model correctly classifies most loans.
Macro Average: 0.94 — The average of precision and recall across both classes.
Weighted Average: 0.99 — This takes into account class imbalance, with higher weights for the larger class (healthy loans).
Key Observations

## Class Imbalance: The dataset contains significantly more healthy loans than high-risk loans. This imbalance likely influenced the model to prioritize predicting healthy loans, resulting in almost perfect accuracy for low-risk loans but slightly reduced performance on high-risk loans.
## Healthy Loans: The model's near-perfect precision and high recall for healthy loans are indicative of the fact that these loans are overrepresented in the dataset, making them easier for the model to identify.
## High-Risk Loans: While the model achieves good recall for high-risk loans (91%), the precision of 85% suggests that it misclassifies some healthy loans as high-risk, which is a critical issue for lenders trying to avoid unnecessary risk.
Optimization Recommendations

## To further improve the model's ability to classify high-risk loans and enhance overall accuracy, we recommend the following steps:

## 1. Addressing Class Imbalance
Given the significant class imbalance, we should focus on improving the model's sensitivity to the minority class (high-risk loans). Possible solutions include:

Oversampling the minority class (high-risk loans) using techniques such as SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic examples and balance the dataset.
Undersampling the majority class (healthy loans) to ensure the model has a more balanced perspective, though this could result in loss of valuable data.

## 2. Model Optimization
Hyperparameter Tuning: Fine-tuning logistic regression hyperparameters (e.g., regularization strength) can improve the model’s performance, particularly for high-risk loans.
Alternative Models: Experiment with more complex algorithms such as Random Forests, Gradient Boosting, or XGBoost, which may perform better on imbalanced datasets and capture more complex relationships in the data.

## 3. Cost-Sensitive Learning
Cost-sensitive Learning: Implementing cost-sensitive learning could help by penalizing the model more for misclassifying high-risk loans. This could improve the model’s precision for high-risk loans and ensure that more attention is given to minimizing false negatives (i.e., loans that are predicted to be healthy but are actually high-risk).

## 4. Feature Engineering
Additional Features: Incorporate additional features that may help improve predictions, such as:
Borrower’s loan history (e.g., past defaults, repayment behavior).
Temporal features like loan duration and recent changes in income or debt.
Geographical Data (if available): The borrower’s location may provide useful insights into their financial behavior or economic stability.

## Conclusion

The logistic regression model demonstrates strong performance, with an overall accuracy of 99%, and is particularly effective at classifying healthy loans. However, the model's precision for high-risk loans (85%) can be improved. Given the importance of correctly identifying high-risk loans to mitigate financial risk, we recommend implementing strategies to address class imbalance, fine-tune the model, and explore alternative algorithms. By focusing on improving performance for high-risk loans, we can further enhance the model's value in peer-to-peer lending and help lenders make better, more informed decisions.
