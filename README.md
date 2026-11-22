Regression and Classification Algorithim CIE  Activities

Name: Harshvardhan Manojkumar Bhosale

Roll Number: AD3121

Class: T.Y. Div: A

This project was completed as part of the Regression and Classification Algorithim CIE  Activities. Faculty reference and Regression and Classification Algorithim learning resources: https://github.com/complexiota

 Repository Overview

This repository contains the completed Jupyter Notebooks/Python scripts, datasets, and reports for the six continuous internal evaluation (CIE) activities for the Regression and Classification Algorithim  course. The activities span across foundational data preparation, supervised learning (regression and classification), unsupervised learning (clustering), model evaluation, and an end-to-end mini-project.
<br>
<br>

## Activity 1: Multiple Linear Regression

### 1. Activity Overview
The goal of Activity 1 is to understand how multiple numerical inputs jointly explain a continuous target variable, verify ordinary least squares (OLS) math manually on a small sample, compare a multiple-feature model with a single-feature baseline, interpret raw vs. standardized coefficients, inspect residuals and multicollinearity (correlation matrix and VIF), and evaluate generalization on a held-out test set.

### 2. Dataset & Source
* **Dataset:** `[Dataset Name, e.g., California Housing Dataset]`
* **Source & Citation:** `[Source Link & Citation, e.g., Scikit-Learn / 1990 U.S. Census via fetch_california_housing]`
* **Dimensions & Format:** `[e.g., 20,640 rows × 9 columns; tabular / CSV / DataFrame]`
* **Target ($y$):** `[Target column name, units, and description, e.g., MedHouseVal in $100,000s]`
* **Input Features ($X_1, \dots, X_p$):** `[List features with units and brief rationale for how each influences y; exclude IDs and personal/confidential data]`

### 3. Work Completed
* **Data Inspection & Cleaning:** Loaded records in Google Colab, checked data types, summary statistics (min, max, range, mean, SD), missing values, and duplicate rows, and documented any cleaning or imputation choices.
* **Exploratory Visualization & Correlation:** Plotted feature-versus-target scatter plots and a correlation heatmap; identified feature–target directions/strengths, inter-feature correlations, and noted that correlation does not imply causation.
* **Manual Calculation Worksheet:** Selected a small sample (`5–8` rows, `2` features), formed design matrix $X$ and target vector $y$, computed $X^T X$ and $X^T y$, solved $(X^T X)b = X^T y$ step by step, wrote $\hat{y} = b_0 + b_1 x_1 + b_2 x_2$, and computed the prediction and residual for one observation.
* **Library Modeling & Scaling:** Split data into train and test sets (`[e.g., 80:20]`), fit `StandardScaler` on training data only when comparing standardized effect sizes, trained a single-feature baseline (strongest correlated feature) and a multiple linear regression model on training data only, and verified the manual sample calculation with a library.
* **Diagnostics, New Prediction & Summary:** Evaluated test predictions, plotted residuals vs. predicted values, residual histogram, and predicted vs. actual (`45°` reference line), computed VIFs, checked regression assumptions, predicted a new record with clear units, and summarized limitations.

### 4. Key Results
* **Manual Sample Verification (`[n]` rows, `[X1, X2]`):**
  * Hand-calculated equation: `y_hat = [b0] + [b1]*x1 + [b2]*x2`
  * Sample observation (`[x1, x2]`, actual `y = [y_actual]`): predicted `y_hat = [y_pred]`, residual `e = [res]`
  * Library verification match: `[Confirmed identical / difference = 0.0000]`
* **Full Model Fitted Equation:** `y_hat = [intercept] + [c1]*[X1] + [c2]*[X2] + ...`
* **Test Set Model Comparison:**

| Metric | Single-Feature Model (`[Top Feature]`) | Multiple-Feature Model (`[p]` Features) |
| :--- | :---: | :---: |
| **MAE** | `[value]` | `[value]` |
| **MSE** | `[value]` | `[value]` |
| **RMSE** | `[value]` | `[value]` |
| **Test $R^2$** | `[value]` | `[value]` |
| **Adjusted $R^2$** | `[value]` | `[value]` |

* **Multicollinearity, Diagnostics & New Record:**
  * High-correlation / high-VIF features identified: `[e.g., AveRooms & AveBedrms — VIF = ...; action: drop/combine/regularize]`
  * Residual & assumption takeaways: `[e.g., residuals centered near 0 with mild heteroscedasticity / ceiling cap]`
  * New example (`[feature values]`): predicted `[value]` `[units]`
  * Key limitations: `[Overlapping features, extra features don't guarantee better test fit, linear model misses non-linearities and omitted factors]`

## Activity 2: Binary Classification (Logistic Regression)

### 1. Activity Overview
The goal of Activity 2 is to build a binary Logistic Regression classifier, understand why linear regression is ill-suited for discrete class labels, examine how the sigmoid function maps log-odds to probabilities in $(0, 1)$, evaluate classification metrics across decision thresholds (`0.3`, `0.5`, `0.7`), compare against a majority-class baseline, and recommend a threshold grounded in error costs.

### 2. Dataset & Source
* **Dataset:** `[Dataset Name, e.g., Breast Cancer Wisconsin (Diagnostic) Dataset]`
* **Source & Citation:** `[Source Link & Citation, e.g., Scikit-Learn / UCI ML Repository via load_breast_cancer]`
* **Dimensions & Format:** `[e.g., 569 rows × 31 columns; tabular / CSV / DataFrame]`
* **Target Column & Class Meanings:** `[Target column name]` where Class `0` = `[meaning of class 0]` and Class `1` = `[meaning of class 1]` (no IDs or personal/confidential data used).
* **Feature Columns:** `[Number and types of numerical/categorical features used]`.

### 3. Work Completed
* **Data Inspection & Class Balance:** Inspected sample records, data types, summary statistics, missing values, duplicates, and label consistency; plotted class counts and assessed whether classes are balanced or imbalanced.
* **Feature Preparation & Stratified Split:** Handled missing values, encoded categorical features where applicable (`[One-Hot / Ordinal / None]`), excluded identifier columns, split data using a stratified `[e.g., 80:20]` train/test split, and fit `StandardScaler` on training data only.
* **Logistic Regression Concept & Training:** Explained why linear regression is unsuitable for class labels, described the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ and decision thresholds, plotted the sigmoid curve, trained Logistic Regression on the training set, and interpreted positive vs. negative coefficients.
* **Threshold Analysis, Evaluation & Decision Plot:** Generated test probabilities and class labels, inspected test rows, compared positive prediction counts and metrics across thresholds `0.3`, `0.5`, and `0.7`, computed confusion matrix components (`TP`, `FP`, `FN`, `TN`), accuracy, precision, recall (sensitivity), specificity, F1-score, classification report, majority-class baseline, ROC curve / ROC-AUC (and PR curve if applicable), train vs. test metrics, and a single-feature probability / decision boundary plot.
* **New Example & Conclusion:** Applied identical preprocessing to a new record to obtain predicted probability and class, summarized the workflow and limitations, and noted that outputs are probability estimates rather than guarantees.

### 4. Key Results
* **Dataset Split & Class Counts:** Train = `[n_train]` (`[c0_train]` class 0, `[c1_train]` class 1); Test = `[n_test]` (`[c0_test]` class 0, `[c1_test]` class 1). Class balance: `[Balanced / Moderately imbalanced / Imbalanced]`.
* **Test Set Performance (Threshold `0.5` vs. Majority Baseline):**
  * Confusion Matrix (`TP`, `FP`, `FN`, `TN`): `TP = [tp]`, `FP = [fp]`, `FN = [fn]`, `TN = [tn]` (`[explain each in domain context]`)
  * **Accuracy:** `[value]` (vs. Majority-Class Baseline: `[baseline_acc]` — accuracy alone can mislead when classes are imbalanced because predicting the majority class ignores minority errors)
  * **Precision:** `[value]` | **Recall (Sensitivity):** `[value]` | **Specificity:** `[value]` | **F1-Score:** `[value]` | **ROC-AUC:** `[value]` (`[ranking ability in plain words]`)
  * **Train vs. Test Check:** Train Acc `[train_acc]` vs. Test Acc `[test_acc]`; Train AUC `[train_auc]` vs. Test AUC `[test_auc]` (`[comment on overfitting/underfitting]`).
* **Threshold Comparison (`0.3`, `0.5`, `0.7`):**

| Threshold | Predicted Class `1` Count | Accuracy | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.3** | `[count]` | `[val]` | `[val]` | `[val]` | `[val]` |
| **0.5** | `[count]` | `[val]` | `[val]` | `[val]` | `[val]` |
| **0.7** | `[count]` | `[val]` | `[val]` | `[val]` | `[val]` |

* **Error Cost, Recommended Threshold & New Record:**
  * Costlier error: `[False Negative / False Positive — explain why]`
  * Recommended threshold: `[e.g., 0.3 / 0.5 / 0.7 — justify precision-recall trade-off]`
  * New record (`[feature summary]`): predicted probability `[prob]` → class `[pred_class]` (`[label]`).
  * Limitations & disclaimer: `[Data/model limitations; predictions are probability-based estimates, not guaranteed outcomes]`.
