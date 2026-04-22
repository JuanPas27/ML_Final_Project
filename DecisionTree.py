import numpy as np
import jax.numpy as jnp

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=7, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.classes = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        self.root = self.make_tree(X, y)

    def entropy(self, y):
        # Shannon Entropy - Σ p(c) log2 p(c).
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def information_gain(self, y, y_left, y_right):
        p = len(y_left) / len(y)
        return self.entropy(y) - p * self.entropy(y_left) - (1 - p) * self.entropy(y_right)

    def best_split(self, X, y):
        best_gain = -1
        best_feat, best_thresh = None, None
        best_splits = None
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for th in thresholds:
                left_mask = X[:, feat] <= th
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                gain = self.information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = th
                    best_splits = (left_mask, right_mask)

        if best_gain <= 0:
            return None
        return best_feat, best_thresh, best_splits

    def make_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        if n_samples < self.min_samples_split or depth >= self.max_depth or len(np.unique(y)) == 1:
            leaf_val = np.argmax(np.bincount(y))
            return Node(value=leaf_val)
        
        # best split
        split = self.best_split(X, y)
        if split is None:
            leaf_val = np.argmax(np.bincount(y))
            return Node(value=leaf_val)

        # split data
        feat, thresh, (left_mask, right_mask) = split

        # make subtrees
        left = self.make_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.make_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(feature_index=feat, threshold=thresh, left=left, right=right)

    def predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        X = np.array(X)
        preds = [self.predict_one(x, self.root) for x in X]
        return jnp.array(preds)