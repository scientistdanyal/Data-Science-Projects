import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from IPython.display import display, Markdown

warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load the Boston housing dataset from a CSV file."""
    df = pd.read_csv(filepath)
    display(Markdown(f"✅ **Data loaded from:** `{filepath}`  \n**Shape:** `{df.shape}`"))
    return df

def show_statistics(df):
    """Display descriptive statistics and data types."""
    display(Markdown("### 📊 Descriptive Statistics"))
    display(df.describe())

    display(Markdown("### 🏷️ Data Types"))
    display(df.dtypes)

    display(Markdown("### ℹ️ Data Info"))
    info_buf = []
    df.info(buf=info_buf.append)
    display(Markdown("```\n" + "\n".join(info_buf) + "\n```"))

def plot_correlation_matrix(df, save_path='outputs/plots/correlation_matrix.png'):
    """Plot and save the correlation heatmap of the dataset."""
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()

    # Ensure the directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    display(Markdown(f"📈 **Correlation matrix saved to:** `{save_path}`"))
    plt.close()

def preprocess_features(df, target_column='MEDV'):
    """Separate features and normalize them. Return normalized X and target y."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_normalized = (X - X.min()) / (X.max() - X.min())
    print(("✅ **Feature normalization completed.**"))
    return X_normalized, y

def preview_data(X_normalized, y):
    """Display the head of features and target data."""
    display(Markdown("### 🔍 Normalized Features Preview"))
    display(X_normalized.head())

    display(Markdown("### 🎯 Target Preview"))
    display(y.head())
