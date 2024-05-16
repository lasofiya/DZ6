# tree.py
import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(y) <= self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return np.mean(y)
        
        best_split = self._find_best_split(X, y)
        if not best_split:
            return np.mean(y)

        left_idx, right_idx = best_split['indices']
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {'feature': best_split['feature'],
                'threshold': best_split['threshold'],
                'left': left_subtree,
                'right': right_subtree}

    def _find_best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None

        best_split = {}
        best_mse = float('inf')

        for feature in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, feature], y)))
            for i in range(1, len(thresholds)):
                if thresholds[i] == thresholds[i - 1]:
                    continue
                left_y, right_y = classes[:i], classes[i:]
                mse = self._calculate_mse(left_y, right_y)
                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        'feature': feature,
                        'threshold': (thresholds[i] + thresholds[i - 1]) / 2,
                        'indices': (np.array(range(i)), np.array(range(i, m)))
                    }

        return best_split

    def _calculate_mse(self, left_y, right_y):
        left_mse = np.mean((left_y - np.mean(left_y))**2)
        right_mse = np.mean((right_y - np.mean(right_y))**2)
        return len(left_y) * left_mse + len(right_y) * right_mse

    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _predict(self, inputs, tree):
        if not isinstance(tree, dict):
            return tree
        if inputs[tree['feature']] < tree['threshold']:
            return self._predict(inputs, tree['left'])
        else:
            return self._predict(inputs, tree['right'])
