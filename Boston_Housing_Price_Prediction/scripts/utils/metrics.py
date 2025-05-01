import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def plot_model_comparison(models, rmse_scores, save_path='outputs/plots/model_comparison_rmse.png'):
    """Generate and save a bar chart comparing RMSE across models."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8,6))
    plt.bar(models, rmse_scores, color=['blue', 'green', 'orange'])
    plt.title('📈 Model Comparison - RMSE')
    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.ylim(0, max(rmse_scores) + 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, score in enumerate(rmse_scores):
        plt.text(i, score + 0.1, f"{score:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Comparison plot saved to: {save_path}")
