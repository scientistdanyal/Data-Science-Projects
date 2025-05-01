import numpy as np

def initialize_weights(n_features):
    return np.zeros(n_features), 0

def predict(X, weights, bias):
    return np.dot(X, weights) + bias

def compute_cost(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, weights, bias, learning_rate, epochs):
    n = len(y)
    cost_history = []

    for epoch in range(epochs):
        y_pred = predict(X, weights, bias)
        dw = -(2/n) * np.dot(X.T, (y - y_pred))
        db = -(2/n) * np.sum(y - y_pred)
        weights -= learning_rate * dw
        bias -= learning_rate * db
        cost_history.append(compute_cost(y, y_pred))
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost_history[-1]:.4f}")
    return weights, bias, cost_history
