from scripts.utils.preprocess import load_data, preprocess_features
from scripts.utils.metrics import rmse, r2_score
from scripts.models.xgboost_manual import XGBoostRegressor

# Load and preprocess data
df = load_data("data/raw/boston.csv")
X, y = preprocess_features(df)

# Manual train/test split
import numpy as np
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X.values[train_idx], X.values[test_idx]
y_train, y_test = y.values[train_idx], y.values[test_idx]

# Train XGBoost model
model = XGBoostRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=5)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\n🏆 Final Tuned XGBoost Model Performance:")
print(f"Final RMSE: {rmse(y_test, y_pred):.4f}")
print(f"Final R² Score: {r2_score(y_test, y_pred):.4f}")
