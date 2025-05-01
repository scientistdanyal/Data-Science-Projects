import numpy as np

# 🌳 Decision Tree Regressor from Scratch
class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def best_split(self, X, y):
        best_mse = float('inf')
        best_idx = None
        best_split = None
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            possible_splits = np.unique(feature_values)
            
            for split in possible_splits:
                left_mask = feature_values <= split
                right_mask = feature_values > split
                
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue  # Skip invalid splits
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                current_mse = (len(left_y) * self.mse(left_y) + len(right_y) * self.mse(right_y)) / n_samples
                
                if current_mse < best_mse:
                    best_mse = current_mse
                    best_idx = feature_idx
                    best_split = split
        
        return best_idx, best_split

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        if (depth >= self.max_depth) or (n_samples < self.min_samples_split):
            return {'value': np.mean(y)}  # Return a leaf node

        best_feature, best_split = self.best_split(X, y)
        
        if best_feature is None:
            return {'value': np.mean(y)}  # No valid split found

        left_mask = X[:, best_feature] <= best_split
        right_mask = X[:, best_feature] > best_split

        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'feature': best_feature,
            'split': best_split,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, tree):
        if 'value' in tree:
            return tree['value']

        feature_value = sample[tree['feature']]
        if feature_value <= tree['split']:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])
