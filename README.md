# Exoplanet-Classification
This project classifies exoplanet candidates from NASA’s Kepler mission as confirmed planets or false positives using machine learning.

---

## 📌 Objective

Classify each planet candidate as either:
- `1`: **CONFIRMED**
- `0`: **FALSE POSITIVE**

using features like orbital period, planet radius, stellar temperature, and more.

---

## 📊 Dataset Overview

- Source: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- Rows: ~9500  
- Columns: 49  
- Target column: `koi_disposition`

---

## 🔧 Steps Followed

### 1. Data Loading
- Used `pandas` to read and inspect the raw CSV.
- Examined column types, null values, and class distributions.

### 2. Data Cleaning (`clean_data.py`)
- Removed:
  - Highly null columns (e.g., error columns with all missing)
  - `CANDIDATE` class entries from the target
- Encoded target: `CONFIRMED = 1`, `FALSE POSITIVE = 0`
- Normalized numeric features using `StandardScaler`

### 3. Exploratory Data Analysis (`3_eda.py`)
- Plotted class counts (bar chart)
- Plotted feature correlation heatmap
- Histograms for key continuous features

### 4. Modeling (`4_model.py`)
- Model: `LogisticRegression`
- Train/test split: 80/20 (stratified)
- Metrics used:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
  - ROC-AUC Score

---

## 📈 Results

- **Accuracy**: 99.67%
- **ROC AUC**: 1.00  
- **Confusion Matrix**:
  - FP: 2
  - FN: 2
- Model was able to **perfectly distinguish** between the two classes in test data.

---

## 📊 Visuals

- **Bar Plot**: Class distribution  
- **Heatmap**: Feature correlations  
- **Confusion Matrix**: Model evaluation  
- **ROC Curve**: Model performance (AUC = 1.00)

---

## 📚 Tech Stack

- Python 3
- pandas
- numpy
- matplotlib, seaborn
- scikit-learn

---

## ✨ Future Work

- Extend to multiclass classification (add `CANDIDATE`)
- Try other models: Random Forest, SVM, XGBoost
- Hyperparameter tuning (GridSearchCV)
- Feature importance & dimensionality reduction (PCA)

---

## 🙋‍♀️ Author

**Soumya Snehal**  
Project for ML Mini Project | Space-AI Exploration  
August 2025

---

