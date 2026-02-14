
# ML Assignment 2 – Classification Models and Streamlit Deployment

------------------------------------------------------------

a. Problem Statement

The objective of this assignment is to implement six different machine learning classification models 
on a publicly available dataset and evaluate their performance using standard evaluation metrics. 
The project also includes building and deploying an interactive Streamlit web application.

------------------------------------------------------------

b. Dataset Description 

Dataset Name: Breast Cancer Wisconsin Dataset  
Source: UCI Machine Learning Repository  
Problem Type: Binary Classification  
Number of Instances: 569  
Number of Features: 30  

Target Variable:
0 – Malignant  
1 – Benign  

This dataset satisfies assignment constraints:
Minimum Feature Size: 12  
Minimum Instance Size: 500  

------------------------------------------------------------

c. Models Used 

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Evaluation Metrics Calculated:
Accuracy  
AUC Score  
Precision  
Recall  
F1 Score  
Matthews Correlation Coefficient (MCC Score)  

------------------------------------------------------------

Model Comparison Table




ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC
-------------------------------------------------------------------
Logistic Regression | 0.9737 | 0.9697 | 0.9722 | 0.9859 | 0.979 | 0.9439
Decision Tree | 0.9474 | 0.944 | 0.9577 | 0.9577 | 0.9577 | 0.888
KNN | 0.9474 | 0.944 | 0.9577 | 0.9577 | 0.9577 | 0.888
Naive Bayes | 0.9649 | 0.9581 | 0.9589 | 0.9859 | 0.9722 | 0.9253
Random Forest | 0.9649 | 0.9581 | 0.9589 | 0.9859 | 0.9722 | 0.9253
XGBoost | 0.9561 | 0.951 | 0.9583 | 0.9718 | 0.965 | 0.9064


------------------------------------------------------------

Observations on Model Performance 

Logistic Regression:
Shows strong overall performance with balanced precision and recall, indicating good linear separability.

Decision Tree:
Provides good interpretability but may slightly overfit compared to ensemble methods.

KNN:
Performance improves significantly after feature scaling and depends on the choice of k.

Naive Bayes:
Computationally efficient and performs well despite independence assumptions.

Random Forest:
Reduces variance compared to a single tree and improves generalization.

XGBoost:
Demonstrates strong predictive capability using gradient boosting.

------------------------------------------------------------



project-folder/
    app.py
    requirements.txt
    README.md
    model/
        train_models.py
    saved_models/
        Logistic Regression.pkl
        Decision Tree.pkl
        KNN.pkl
        Naive Bayes.pkl
        Random Forest.pkl
        XGBoost.pkl
        scaler.pkl
        model_results.csv

------------------------------------------------------------

Streamlit Application Features 

Dataset upload option (CSV – test data only)  
Model selection dropdown  
Display of evaluation metrics  
Confusion matrix  
Classification report  
Download predictions as CSV  

------------------------------------------------------------

How to Run the Project

1. pip install -r requirements.txt  
2. python model/train_models.py  
3. streamlit run app.py  

------------------------------------------------------------

Deployment Instructions

1. Go to https://streamlit.io/cloud  
2. Sign in using GitHub  
3. Click New App  
4. Select your repository  
5. Choose branch main  
6. Select app.py  
7. Click Deploy  



