# Machine Learning - Assignment 2

Student Name: Sneha Singh
BITS ID: 2025AA05176
Programme: M.Tech (AIML/DSE)
Course: Machine Learning
University: Birla Institute of Technology and Science (BITS Pilani)


**Live Deployment Links**
**GitHub Repository**:
https://github.com/2025aa05176/MLAssignment2

**Live Streamlit Application:**
https://mlassignment2-4utyr2xfr6k4io3sndvb8g.streamlit.app/



------------------------------------------------------------------------

## 1. Problem Statement

Implement multiple classification models on a public dataset, evaluate
them using standard evaluation metrics, and deploy an interactive
Streamlit web application to demonstrate model selection and prediction.

------------------------------------------------------------------------

## 2. Dataset Description

  Attribute             Details
  --------------------- --------------------------------------
  Dataset Name          Breast Cancer Wisconsin (Diagnostic)
  Source                UCI Repository (via sklearn)
  Total Instances       569
  Total Features        30 Numerical Features
  Classification Type   Binary Classification
  Target Classes        Malignant / Benign

The dataset satisfies the minimum requirement of at least 12 features
and 500 instances.

------------------------------------------------------------------------

## 3. Models Used

The following six classification models were implemented on the same
dataset:

1.  Logistic Regression\
2.  Decision Tree Classifier\
3.  K-Nearest Neighbor (kNN)\
4.  Naive Bayes Classifier (Gaussian)\
5.  Random Forest (Ensemble Model)\
6.  XGBoost (Ensemble Model)

------------------------------------------------------------------------

## 4. Evaluation Metrics Used

For each model, the following metrics were calculated:

-   Accuracy\
-   AUC Score\
-   Precision\
-   Recall\
-   F1 Score\
-   Matthews Correlation Coefficient (MCC)

------------------------------------------------------------------------

## 5. Model Performance Comparison

  -------------------------------------------------------------------------------
  ML Model Name     Accuracy    AUC      Precision    Recall    F1       MCC
  ----------------- ----------- -------- ------------ --------- -------- --------
  Logistic          0.9649      0.9954   0.9595       0.9861    0.9726   0.9120
  Regression                                                             

  Decision Tree     0.9123      0.9157   0.9559       0.9028    0.9286   0.8250

  kNN               0.9123      0.9559   0.9429       0.9167    0.9296   0.8300

  Naive Bayes       0.9737      0.9984   0.9595       1.0000    0.9793   0.9447

  Random Forest     0.9649      0.9921   0.9589       0.9859    0.9722   0.9253
  (Ensemble)                                                             

  XGBoost           0.9561      0.9908   0.9583       0.9718    0.9650   0.9064
  (Ensemble)                                                             
  -------------------------------------------------------------------------------

------------------------------------------------------------------------

## 6. Observations on Model Performance

  -----------------------------------------------------------------------
  ML Model Name        Observation about Model Performance
  -------------------- --------------------------------------------------
  Logistic Regression  Strong baseline model with excellent AUC and
                       balanced precision-recall performance.

  Decision Tree        Interpretable but slightly lower AUC; prone to
                       overfitting and variance.

  kNN                  High recall and competitive accuracy; sensitive to
                       feature scaling.

  Naive Bayes          Achieved highest overall performance with strong
                       recall and F1 score.

  Random Forest        More robust and stable compared to single Decision
  (Ensemble)           Tree.

  XGBoost (Ensemble)   Balanced performance across metrics with efficient
                       boosting optimization.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 7. Streamlit Application Features

The deployed Streamlit application includes:

-   Dataset upload option (CSV)\
-   Model selection dropdown\
-   Display of evaluation metrics\
-   Confusion matrix visualization\
-   Classification report (tabular format)

------------------------------------------------------------------------

## 8. Project Repository Structure

    project-folder/
    │-- app.py
    │-- requirements.txt
    │-- README.md
    │-- model/

------------------------------------------------------------------------


## Conclusion

All six classification models were implemented and evaluated using
standardized metrics.\
Ensemble methods and Naive Bayes achieved strong overall performance,
while Logistic Regression provided a reliable and interpretable
baseline.

The project demonstrates end-to-end machine learning workflow including
modeling, evaluation, and deployment using Streamlit.
