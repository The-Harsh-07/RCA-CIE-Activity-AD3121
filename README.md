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
Activity 1 builds a multiple linear regression model to predict a continuous target from several numerical input features. It pairs a step-by-step manual normal-equation calculation on a small sample with a full library implementation in Google Colab, compares a single-feature baseline against the multi-feature model, interprets raw and standardized coefficients, checks multicollinearity via correlation and Variance Inflation Factor (VIF), inspects residual plots, and evaluates generalization on an unseen test split.

### 2. Dataset & Source
* **Dataset:** California Housing Dataset
* **Source & Citation:** Scikit-Learn (`sklearn.datasets.fetch_california_housing`), derived from the 1990 U.S. Census via StatLib (`https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html`)
* **Dimensions & Format:** `20,640` rows × `9` columns (loaded as a tabular `pandas.DataFrame`; no personal/confidential data and no ID column)
* **Target ($y$):** `MedHouseVal` (Median House Value in `$100,000s`, capped around `5.00001` in the dataset)
* **Input Features ($X_1, \dots, X_8$):**
  * `MedInc` (`$10,000s`): Higher local median income increases housing purchasing power and neighborhood property values.
  * `HouseAge` (`years`): Median age of housing stock reflects neighborhood maturity, central location, or maintenance needs.
  * `AveRooms` (`rooms/household`): Larger average room counts generally indicate bigger homes with higher value.
  * `AveBedrms` (`bedrooms/household`): Reflects sleeping capacity and layout, closely related to total rooms.
  * `Population` (`persons`): Block-group population captures neighborhood scale and density.
  * `AveOccup` (`persons/household`): Average household size reflects occupancy density and housing type.
  * `Latitude` (`degrees N`): Captures north–south geographic variation across California.
  * `Longitude` (`degrees W`): Captures east–west proximity to coastal employment centers versus inland areas.

### 3. Work Completed
* **Data Inspection & Cleaning:** Loaded the dataset in Colab, checked data types (`float64` across all columns), computed summary statistics (`min`, `max`, `range`, `mean`, `std`, median) with units, verified `0` missing values and `0` duplicate rows in the standard fetch, and wrote median-imputation / non-positive value checks for general robustness.
* **Exploratory Visualization & Correlation:** Generated scatter plots of `MedHouseVal` against each input feature and a labeled correlation matrix heatmap. `MedInc` showed the strongest positive correlation with `MedHouseVal` (`r ≈ +0.69`), while `AveRooms` and `AveBedrms` showed strong positive correlation with each other (`r ≈ +0.85`), signaling potential multicollinearity. Noted that correlation does not prove causation.
* **Manual Calculation Worksheet:** Selected a 6-row sample with two features (`x1` and `x2`) and target `y`, constructed design matrix `X` (column of `1`s, `x1`, `x2`) and target vector `y`, computed `X^T X` and `X^T y`, solved `(X^T X)b = X^T y` via elimination for `b0 = 2`, `b1 = 3`, `b2 = 1`, and verified observation 4 (`x1 = 4, x2 = 2, y = 16` -> predicted `16`, residual `0`).
* **Library Modeling & Scaling:** Split the `20,640` rows `80:20` into `16,512` training rows and `4,128` test rows (`random_state=42`). Fit `StandardScaler` strictly on `X_train` to obtain standardized coefficients for comparing relative effect sizes without altering OLS predictions. Trained a single-feature baseline (`MedInc`) and an 8-feature multiple linear regression model on the training set only, and confirmed the manual sample parameters matched `scikit-learn` (`0.000000` difference).
* **Diagnostics, New Prediction & Summary:** Generated test predictions, plotted residuals vs. predicted values, a residual histogram, and predicted vs. actual values against a `45°` reference line, computed VIFs, reviewed linear regression assumptions, predicted `MedHouseVal` for a new record, and summarized model limitations.

### 4. Key Results
* **Manual Sample Verification (`6` rows, `x1` and `x2`):**
  * Hand-calculated equation: `y_hat = 2.0000 + 3.0000*x1 + 1.0000*x2`
  * Sample observation (`x1 = 4, x2 = 2`, actual `y = 16`): predicted `y_hat = 16.0000`, residual `e = 0.0000`
  * Library verification match: Exact match (`0.000000` difference)
* **Full Model Fitted Equation (Illustrative Sample Run):**
  `MedHouseVal = -36.8500 + 0.4360*MedInc + 0.0095*HouseAge - 0.1070*AveRooms + 0.6450*AveBedrms - 0.000004*Population - 0.0038*AveOccup - 0.4210*Latitude - 0.4340*Longitude`
  *(In `$100,000s`; note that setting all features to `0` simultaneously—such as `0°` latitude/longitude or `$0` income—is outside realistic California data bounds, so the raw intercept is a mathematical baseline rather than a meaningful home price.)*
* **Test Set Model Comparison (`n_test = 4,128`):**

| Metric | Single-Feature Model (`MedInc`) | Multiple-Feature Model (All `8` Features) |
| :--- | :---: | :---: |
| **MAE** | `0.6299` (`$62,990`) | `0.5332` (`$53,320`) |
| **MSE** | `0.7091` | `0.5559` |
| **RMSE** | `0.8421` (`$84,210`) | `0.7456` (`$74,560`) |
| **Test $R^2$** | `0.4589` | `0.5758` |
| **Adjusted $R^2$** | `0.4588` | `0.5750` |

  * **Complexity Justification:** Adding the remaining 7 spatial and structural features lowers test RMSE from `0.8421` to `0.7456` (`$100,000s`) and raises Adjusted $R^2$ from `0.4588` to `0.5750`, justifying the added complexity as a baseline linear model while leaving room for non-linear models.
* **Multicollinearity, Diagnostics & New Record:**
  * **Multicollinearity (VIF):** `AveRooms` (`VIF ≈ 8.3`) and `AveBedrms` (`VIF ≈ 6.9`), along with `Latitude` and `Longitude` (`VIF > 9`), show substantial overlap; practical remedies include combining bedroom-to-room ratios (`AveBedrms / AveRooms`), dropping redundant variables, or using regularized regression.
  * **Residual & Assumption Check:** Residuals center near `0` and are roughly symmetric in the middle, though spread varies across predicted values (mild heteroscedasticity) and `MedHouseVal` has a ceiling cap near `5.0`, causing under-prediction at high actual values.
  * **New Example (`MedInc = 5.2`, `HouseAge = 25.0`, `AveRooms = 5.5`, `AveBedrms = 1.1`, `Population = 1200`, `AveOccup = 2.8`, `Latitude = 34.05`, `Longitude = -118.25`):** Predicted `MedHouseVal ≈ 2.46` (`$100,000s`), or about `$246,000`.
  * **Limitations:** Correlated predictors share information, adding more features does not always improve generalization, and a linear equation misses curved relationships, non-linear geography, and omitted factors such as local school quality or interest rates.

## Activity 2: Binary Classification (Logistic Regression)

### 1. Activity Overview
Activity 2 builds a binary Logistic Regression model to predict a two-class target from numerical features. It explains why linear regression is unsuitable for class labels, demonstrates how the sigmoid function converts linear log-odds into probabilities between `0` and `1`, compares model performance against a majority-class baseline, evaluates confusion-matrix metrics across decision thresholds (`0.3`, `0.5`, `0.7`), and recommends a threshold based on error costs.

### 2. Dataset & Source
* **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Source & Citation:** Scikit-Learn (`sklearn.datasets.load_breast_cancer`), originally from the UCI Machine Learning Repository (`https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html`)
* **Dimensions & Format:** `569` rows × `31` columns (`30` continuous numerical cell-nucleus features + `1` binary target; no personal/confidential records and no ID column)
* **Target Column & Class Meanings:** `target` (`0` = **malignant** / cancerous tumor; `1` = **benign** / non-cancerous tumor)
* **Feature Columns:** `30` continuous measurements (`mean`, `error`, and `worst` statistics for `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry`, and `fractal dimension`).

### 3. Work Completed
* **Data Inspection & Class Balance:** Displayed sample rows, data types (`float64` features, `int64` target), and summary statistics; verified `0` missing values, `0` duplicate rows, and clean `{0, 1}` target values. Plotted class counts (`357` benign / `62.7%` vs. `212` malignant / `37.3%`), noting a moderate class imbalance.
* **Feature Preparation & Stratified Split:** Confirmed all `30` features are numerical (no categorical encoding required; no ID column used). Split data `80:20` using `stratify=y` (`455` train rows, `114` test rows) and fit `StandardScaler` strictly on `X_train` to aid optimizer convergence and keep feature scales comparable.
* **Logistic Regression Concept & Training:** Explained why unbounded linear regression is poorly suited for binary class labels, defined the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ and decision thresholds, plotted the sigmoid curve, trained `LogisticRegression(max_iter=1000)` on `X_train_scaled`, and interpreted positive vs. negative coefficients.
* **Threshold Analysis, Evaluation & Decision Plot:** Generated test probabilities and class labels, inspected sample test predictions, compared positive prediction counts and metrics across thresholds `0.3`, `0.5`, and `0.7`, computed the confusion matrix (`TP`, `FP`, `FN`, `TN`), accuracy, precision, recall (sensitivity), specificity, F1-score, full classification report, majority-class baseline, ROC curve / ROC-AUC, precision–recall curve, train vs. test metrics, and a single-feature probability plot (`worst concave points`).
* **New Example & Conclusion:** Applied the fitted scaler and classifier to a new record to obtain its predicted probability and class, summarized the workflow and limitations, and emphasized that predictions are probability estimates rather than guarantees.

### 4. Key Results
* **Dataset Split & Class Counts:**
  * **Train (`455`):** `170` malignant (`0`) and `285` benign (`1`)
  * **Test (`114`):** `42` malignant (`0`) and `72` benign (`1`)
  * **Class Balance:** Moderately imbalanced (`~62.7%` class `1` vs. `~37.3%` class `0`).
* **Test Set Performance at Threshold `0.5` (Illustrative Run, `n_test = 114`):**
  * **Confusion Matrix (`TP`, `FP`, `FN`, `TN`):** `TP = 71`, `FP = 2`, `FN = 1`, `TN = 40` (with class `1` treated as positive in standard formulas: `71` benign correctly predicted benign, `2` malignant misclassified as benign, `1` benign misclassified as malignant, `40` malignant correctly predicted malignant).
  * **Accuracy:** `0.9737` (`97.37%`) vs. **Majority-Class Baseline Accuracy:** `0.6316` (`63.16%`, always predicting benign). Relying on accuracy alone on imbalanced datasets can be misleading because a majority-class classifier can look moderately accurate while completely failing on the smaller class.
  * **Precision (class `1`):** `0.9726` (`TP / (TP + FP)`)
  * **Recall / Sensitivity (class `1`):** `0.9861` (`TP / (TP + FN)`)
  * **Specificity (class `0`):** `0.9524` (`TN / (TN + FP)`)
  * **F1-Score (class `1`):** `0.9793` (`2 * Precision * Recall / (Precision + Recall)`)
  * **ROC-AUC:** `0.9957` (indicates strong ranking ability—the model assigns a higher score to a randomly chosen positive record than a randomly chosen negative record nearly `99.6%` of the time).
  * **Train vs. Test Check:** Train Accuracy `~0.9868` vs. Test Accuracy `0.9737` (and comparable ROC-AUC), indicating strong generalization without severe overfitting or underfitting.
* **Threshold Comparison (`0.3`, `0.5`, `0.7`):**

| Threshold | Predicted Class `1` Count | Accuracy | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.3** | `75` | `0.9649` | `0.9467` | `0.9861` | `0.9660` |
| **0.5** | `73` | `0.9737` | `0.9726` | `0.9861` | `0.9793` |
| **0.7** | `70` | `0.9649` | `0.9857` | `0.9583` | `0.9718` |

  * **Trade-Off & Error Cost:** Lowering the threshold increases positive predictions and recall while tending to lower precision; raising the threshold increases precision while lowering recall. In medical screening, **missing a malignant tumor** is clinically far more costly than a false alarm requiring follow-up testing—so if malignant is coded as the positive target event, a lower threshold (or explicit clinical review of borderline probabilities) is preferred to maximize sensitivity, whereas at `0.5` on this dataset both precision and recall remain high (`> 0.97`).
  * **New Record (`median feature values`):** Predicted probability `P(benign = 1) ≈ 0.96` -> predicted class `1` (`benign`).
  * **Limitations & Disclaimer:** Results depend on historical diagnostic data and feature quality, and a model output is a **probability-based estimate**, not a guaranteed clinical diagnosis.
