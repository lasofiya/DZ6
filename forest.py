import numpy as np
from tree import DecisionTreeRegressor
from joblib import Parallel, delayed


class RandomForestRegressor:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features='sqrt', n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.trees = []

    def fit(self, X, y):
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_tree)(X, y) for _ in range(self.n_estimators))

    def _fit_single_tree(self, X, y):
        sample_X, sample_y = self._bootstrap_sample(X, y)
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        tree.fit(sample_X, sample_y)
        return tree

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        tree_predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.trees)
        return np.mean(tree_predictions, axis=0)
