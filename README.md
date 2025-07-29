# 🌳 Decision Tree Classifier with GridSearchCV & Manual Accuracy Calculation

This project showcases a **Decision Tree Classification** model built using `scikit-learn`, where:

- ✅ Hyperparameters are tuned using **GridSearchCV**
- ✅ Accuracy is evaluated **manually** and with `sklearn.metrics`
- ✅ Training Accuracy: **0.98**
- ✅ Testing Accuracy: **0.89**

---

## 🧠 Algorithm Overview

A **Decision Tree** is a supervised learning model that splits data recursively into branches to predict an outcome. It selects splits based on **Gini impurity** or **Information Gain (Entropy)** to make the best decisions at each node.

> Decision Trees are interpretable and flexible, but they can easily overfit — so tuning parameters like `max_depth`, `min_samples_split`, and `min_samples_leaf` is essential.

---

## ⚙️ Libraries Used

- Python 3.x
- `pandas`, `numpy`
- `scikit-learn` (for model building and evaluation)
- `matplotlib`, `seaborn` *(optional for visualization)*

---

## 📌 Steps Performed

1. **Data Preparation**
   - Load and clean dataset
   - Perform train-test split

2. **GridSearchCV for Hyperparameter Tuning**
   ```python
   from sklearn.model_selection import GridSearchCV

   param_grid = {
       'criterion': ['gini', 'entropy'],
       'max_depth': [3, 5, 10, None],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 2, 4]
   }

   grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
   grid.fit(X_train, y_train)

   best_model = grid.best_estimator_
