# 📊 Telco Customer Churn Prediction  
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow.svg)](https://scikit-learn.org/)  
[![RandomForest](https://img.shields.io/badge/RandomForest-ML-green.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
[![AWS Ready](https://img.shields.io/badge/AWS-Deployment%20Ready-orange.svg)](https://aws.amazon.com/)

---

## 🚀 Project Overview

**Churn prediction** enables businesses to identify which customers are at risk of leaving ("churn")—helping reduce revenue loss and improve retention.  
This project analyzes the Telco Customer Churn dataset and builds an end-to-end pipeline, culminating in a highly accurate Random Forest model, ready for deployment on AWS.

- **For Beginners:** Step-by-step walkthrough covering data cleaning, feature engineering, class balancing, model building, evaluation, and export for production.
- **For Recruiters:** Demonstrates strong practical ML engineering skills: robust preprocessing, handling imbalance, advanced model selection and tuning, and deployment readiness.

---

## 📂 Dataset Overview

The dataset comes from a fictional telco company and contains **7043 customer records** with a rich set of features:

- **Customer demographics:** Age, gender, location, senior citizen status, dependents, etc.
- **Account information:** Tenure, contract type, payment method, monthly/total charges, etc.
- **Services signed up for:** Phone service, internet plans, add-ons like security, backup, device protection, streaming, etc.
- **Churn details:** Churn label/value, churn reason, lifetime value, and predictive churn score.

**Dataset link:**  
[Telco Customer Churn IBM Dataset on Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data)
---

## 📝 Approach & Workflow

1. **Data Cleaning & Preprocessing**
   - Missing value handling  
   - Categorical variable encoding (Label/One-Hot)  
   - Data leakage prevention  
   - Feature scaling

2. **Feature Engineering**
   - Creation of new informative variables  
   - Removal of redundant columns

3. **Class Imbalance Handling**
   - Upsampling of minority class to address imbalance

4. **Model Training & Selection**
   - Multiple algorithms tested; **Random Forest Classifier** showed best results
   - Construction of reproducible scikit-learn Pipelines

5. **Hyperparameter Tuning**
   - Used `RandomizedSearchCV` and `GridSearchCV` for Random Forest optimization

6. **Model Evaluation**
   - Performance metrics:
     - **Accuracy:** `0.92`
     - **ROC-AUC:** `0.92`
     - **Precision, Recall, F1-score:** See table below
     - **Confusion Matrix:**  
       ```
       [[1351  167]
        [  74 1510]]
       ```

7. **Deployment Readiness**
   - Model exported to `.pkl` format  
   - Ready for deployment on **AWS**

---

## 🎯 Results & Metrics

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
|   0   |   0.95    |  0.89  |   0.92   |  1518   |
|   1   |   0.90    |  0.95  |   0.93   |  1584   |

**Overall metrics:**  
- **Accuracy:** `0.922`  
- **ROC-AUC:** `0.922`  
- **Macro Avg:** Precision: `0.92`, Recall: `0.92`, F1: `0.92`  
- **Weighted Avg:** Precision: `0.92`, Recall: `0.92`, F1: `0.92`  

---

## 💻 Tech Stack

| Tool            | Purpose                       |
|-----------------|------------------------------|
| Python          | Programming Language          |
| pandas          | Data wrangling                |
| scikit-learn    | ML algorithms & pipelines     |
| RandomForest    | Model of choice               |
| matplotlib / seaborn | Data visualization     |
| Jupyter / Colab | Interactive notebooks         |
| AWS             | Model deployment              |

---

## 🛠️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Dibyo10/Customer-Churn-Analysis-IBM.git
cd Customer-Churn-Analysis-IBM

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook
# Or open in Google Colab by uploading the notebook
```

- Open the notebook (`.ipynb` file), run cells sequentially, and follow along!

---

## 🗂️ Model Export & AWS Deployment

Trained model is saved in `.pkl` format for production use.  
Example:

```python
import pickle

# Save model
with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('best_rf_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

**Deployment:**  
- The exported `.pkl` file is ready to be deployed on AWS (e.g., via SageMaker, Lambda, or EC2 for serving predictions).

---

## 🌱 Future Improvements

- Test deep learning models (ANN, etc.)
- Deploy a REST API for predictions using **FastAPI** or build a dashboard with **Streamlit**
- Integrate with real business data streams for live inference
- Add model monitoring and automated retraining

---

## 💡 Why Churn Prediction Matters

> "Predicting customer churn empowers businesses to proactively retain valuable customers, reducing revenue loss and boosting growth."

---

## 🙋 About Me

I built this project to showcase robust ML engineering skills:
- Advanced data preprocessing (handling missing/categorical data, leakage prevention)
- Correctly handling class imbalance (upsampling)
- Experience with model selection and hyperparameter tuning
- Deployment-ready pipeline and AWS integration

**Connect with me on [GitHub](https://github.com/Dibyo10)!**

---

## 🏁 Get Started Now!

```
Clone, run, explore, and build upon this project.  
Whether you're a student learning ML or a recruiter seeking ML engineering talent—this repo demonstrates robust skills and real-world impact.
```

---
