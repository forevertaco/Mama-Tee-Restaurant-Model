# Mama-Tee-Restaurant-Model
## By Judah Alonge
Building a regression task to predict the amount of tips
---

````markdown
# 🧠 MAMA Tee ML
**Author:** Judah Alonge  

A simple **machine learning regression project** that predicts restaurant tips based on customer and order features using **Linear Regression**, **Decision Tree**, and **Random Forest** models.

---

## 📦 Project Overview

This project demonstrates:
- Data loading and exploration using **Pandas**
- Feature preprocessing with **One-Hot Encoding**
- Model training using **Scikit-learn**
- Model evaluation and comparison using common regression metrics

---

## 🧰 Technologies Used
- Python 🐍  
- Pandas  
- NumPy  
- Scikit-learn  

---

## 📂 Dataset

**File:** `data/tips.csv`  
Example preview:

| total_bill | tip   | gender | smoker | day | time  | size |
|-------------|-------|--------|--------|-----|-------|------|
| 2125.50 | 360.79 | Male | No | Thur | Lunch | 1 |
| 2727.18 | 259.42 | Female | No | Sun | Dinner | 5 |
| 1066.02 | 274.68 | Female | Yes | Thur | Dinner | 4 |
| 3493.45 | 337.90 | Female | No | Sun | Dinner | 1 |
| 3470.56 | 567.89 | Male | Yes | Sun | Lunch | 6 |

**Shape:** `(744, 7)`  
**No missing values found.**

---

## 🧭 Data Exploration

```python
df.info()
````

Output:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 744 entries, 0 to 743
Data columns (total 7 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   total_bill  744 non-null    float64
 1   tip         744 non-null    float64
 2   gender      744 non-null    object 
 3   smoker      744 non-null    object 
 4   day         744 non-null    object 
 5   time        744 non-null    object 
 6   size        744 non-null    int64  
```

---

## 🔧 Data Preprocessing

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

y = df['tip']
X = df.drop('tip', axis=1)

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🤖 Model Training

### 1️⃣ Linear Regression

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

### 2️⃣ Decision Tree

```python
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
```

### 3️⃣ Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

---

## 📈 Model Evaluation

Metrics used:

* **MAE** (Mean Absolute Error)
* **RMSE** (Root Mean Square Error)
* **R² Score**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "MAE": [
        mean_absolute_error(y_test, y_pred),
        mean_absolute_error(y_test, y_pred_dt),
        mean_absolute_error(y_test, y_pred_rf)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred)),
        np.sqrt(mean_squared_error(y_test, y_pred_dt)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf))
    ],
    "R2 Score": [
        r2_score(y_test, y_pred),
        r2_score(y_test, y_pred_dt),
        r2_score(y_test, y_pred_rf)
    ]
})

print(results)
```

### 🧮 Model Comparison Table

| Model             | MAE    | RMSE   | R² Score |
| ----------------- | ------ | ------ | -------- |
| Linear Regression | 129.98 | 163.89 | 0.0138   |
| Decision Tree     | 158.04 | 205.36 | -0.5485  |
| Random Forest     | 125.81 | 160.57 | 0.0533   |

---

## 🏁 Results Summary

* **Best model:** ✅ Random Forest
* **Lowest MAE:** 125.81
* **Highest R²:** 0.0533 (still low — potential for model improvement with feature engineering)

---

## 🚀 Future Improvements

* Feature scaling and outlier removal
* Hyperparameter tuning (GridSearchCV)
* Experiment with Gradient Boosting or XGBoost
* Visualizations for model insights

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

👨‍💻 *Created by [Judah Alonge](https://github.com/forevertaco)*

```

---


