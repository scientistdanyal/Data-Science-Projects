import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scripts.utils.preprocess import load_data, preprocess_features
from scripts.utils.metrics import rmse, r2_score
from scripts.models.linear_regression import initialize_weights, gradient_descent, predict

# Load and preprocess data
df = load_data("data/raw/boston.csv")
X, y = preprocess_features(df)

# Initialize weights
weights, bias = initialize_weights(X.shape[1])

# Train model
weights, bias, cost_history = gradient_descent(X.values, y.values, weights, bias, learning_rate=0.01, epochs=1000)

# Predict
y_pred = predict(X.values, weights, bias)

# Evaluate
print("\n📈 Linear Regression Performance:")
print(f"RMSE: {rmse(y.values, y_pred):.4f}")
print(f"R² Score: {r2_score(y.values, y_pred):.4f}")

# 📸 Save prediction plot
output_dir = Path("outputs/plots")
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8,6))
plt.scatter(y, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.grid(True)

# Save plot
plot_path = output_dir / "linear_regression_predictions.png"
plt.savefig(plot_path)
plt.close()

print(f"📊 Plot saved to: {plot_path}")
