# 🏠 Boston Housing Price Prediction

A complete end-to-end machine learning project where we predict house prices using the classic Boston Housing dataset. The project demonstrates hands-on model building **from scratch** — without using high-level libraries like `scikit-learn` for model fitting.

---

## 📊 Dataset

- **Source**: Kaggle - The Boston House Price Dataset  
- **Target Variable**: `MEDV` — Median value of owner-occupied homes in \$1000s  
- **Features**: 13 numerical and categorical columns including crime rate, number of rooms, tax rate, etc.

---

## 🧠 Project Approach

1. Load and inspect data
2. Normalize features using Min-Max scaling
3. Split data into train/test sets manually (80/20)
4. Build models from scratch:
   - Linear Regression
   - Random Forest
   - XGBoost
5. Evaluate performance using RMSE & R² Score
6. Compare models visually
7. Save the best model

---

## 📐 Linear Regression - Formulas

**Basic Equation**:  
\[
\hat{y} = wX + b
\]

**Cost Function (Mean Squared Error)**:  
\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

**Gradient Descent Update Rules**:  
\[
w = w - \alpha \cdot \frac{\partial MSE}{\partial w}  
\quad\quad  
b = b - \alpha \cdot \frac{\partial MSE}{\partial b}
\]

---

## 🔁 Vectorization Strategy

- Predictions: `np.dot(X, weights)`
- Gradients and MSE calculated using matrix operations
- Entire model trained without loops (except epochs)

---

## 🔢 Normalization

All features were scaled using **Min-Max Normalization**:

\[
X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
\]

This ensures all features lie between 0 and 1, improving model performance.

---

## 🤖 Models Implemented

### 🔹 Linear Regression (from scratch)
- Gradient Descent on MSE
- Manual updates to weights and bias

### 🔹 Random Forest (from scratch)
- Bootstrap sampling per tree
- Custom decision tree implementation
- Prediction = average of trees

### 🔹 XGBoost (from scratch)
- Sequential boosting on residuals
- Trees trained stage-wise
- Controlled using learning rate

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE   | Root Mean Squared Error - average prediction error magnitude |
| R²     | Coefficient of Determination - how well the model explains variance |

---

## 🏆 Final Results

| Model             | RMSE  | R² Score |
|------------------|-------|----------|
| Linear Regression| 5.78  | 0.60     |
| Random Forest     | 3.50  | 0.83     |
| XGBoost           | 3.00  | 0.87     |

✅ **Best Model: XGBoost**



