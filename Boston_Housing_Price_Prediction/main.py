import numpy as np
import pickle
from scripts.utils import preprocess
from scripts.utils.metrics import rmse, r2_score, plot_model_comparison
from scripts.models.linear_regression import initialize_weights, gradient_descent, predict as lin_predict
from scripts.models.random_forest import RandomForestRegressor
from scripts.models.XGBoostRegressor import XGBoostRegressor

# Step 1: Load and preprocess data
df = preprocess.load_data("data/raw/boston.csv")
X, y = preprocess.preprocess_features(df)

# Step 2: Train-test split
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split = int(0.8 * len(indices))
X_train, X_test = X.values[indices[:split]], X.values[indices[split:]]
y_train, y_test = y.values[indices[:split]], y.values[indices[split:]]

# Step 3: Train Linear Regression
w, b = initialize_weights(X_train.shape[1])
w, b, _ = gradient_descent(X_train, y_train, w, b, learning_rate=0.01, epochs=1000)
y_pred_lr = lin_predict(X_test, w, b)
rmse_lr = rmse(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Step 4: Train Random Forest
rf = RandomForestRegressor(n_estimators=10, max_depth=7, min_samples_split=2)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse_rf = rmse(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Step 5: Train XGBoost
xgb = XGBoostRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=5)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
rmse_xgb = rmse(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Step 6: Compare all
models = ['Linear Regression', 'Random Forest', 'XGBoost']
rmse_scores = [rmse_lr, rmse_rf, rmse_xgb]
r2_scores = [r2_lr, r2_rf, r2_xgb]

plot_model_comparison(models, rmse_scores)

# Step 7: Identify best model
best_index = np.argmin(rmse_scores)
best_model_name = models[best_index]
print(f"\n✅ Best Model: {best_model_name} (RMSE = {rmse_scores[best_index]:.4f})")

# Step 8: Save best model (Random Forest / XGBoost only – Linear can't be pickled easily)
if best_model_name == 'Random Forest':
    best_model = rf
elif best_model_name == 'XGBoost':
    best_model = xgb
else:
    best_model = {'weights': w, 'bias': b}

import os
os.makedirs("models", exist_ok=True)

if best_model_name == 'Linear Regression':
    with open("models/linear_regression_params.pkl", "wb") as f:
        pickle.dump(best_model, f)
else:
    with open(f"models/{best_model_name.lower().replace(' ', '_')}.pkl", "wb") as f:
        pickle.dump(best_model, f)
