# 🧬 Chronic Kidney Disease (CKD) Prediction

A machine learning-based diagnostic tool to predict the presence of chronic kidney disease (CKD) using clinical patient data. This project leverages classification algorithms to assist early detection, which can significantly improve treatment outcomes.

---

## 📊 Overview

- ✅ Preprocessed and cleaned clinical dataset (missing values, data types, normalization)
- ✅ Trained multiple classification models to predict CKD status
- ✅ Evaluated model performance using accuracy, confusion matrix, and classification report
- ✅ Visualized insights to understand important features and model behavior

---

## 🧠 Problem Statement

Chronic Kidney Disease is a condition characterized by a gradual loss of kidney function over time. Early detection is critical. Given a patient's clinical data (e.g., blood pressure, albumin levels, hemoglobin), the goal is to build a classifier that predicts whether the patient has CKD.

---

## 📁 Dataset

- **Source:** [UCI Machine Learning Repository - Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)
- **Instances:** 400
- **Features:** 25 attributes including age, blood pressure, specific gravity, blood glucose, albumin, hemoglobin, and more
- **Target:** `"classification"` — binary label (`ckd` or `notckd`)

---

## 🧰 Tools & Technologies

- **Python**
- **Pandas & NumPy** – data handling
- **Matplotlib & Seaborn** – data visualization
- **Scikit-learn** – ML models & evaluation
- **Jupyter Notebook** – development environment

---

## 🧼 Data Preprocessing

- Handled missing values with imputation
- Converted categorical features to numeric (label encoding)
- Normalized numerical features
- Removed irrelevant or highly missing columns (if any)

---

## 🤖 Models Used

| Model                | Accuracy (%) |
|---------------------|--------------|
| Logistic Regression | 97%          |
| K-Nearest Neighbors | 95%          |
| Decision Tree       | 96%          |
| Random Forest       | 98%          |

> You can experiment with different test sizes, scalers, or hyperparameters to improve further.

---

## 📈 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Visualization of feature importance (for tree-based models)

---

