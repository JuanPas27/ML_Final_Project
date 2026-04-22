import jax.numpy as jnp
from collections import namedtuple

Node = namedtuple('Node', ['feature_idx', 'threshold', 'left', 'right', 'value'])

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.classes = None

    def _entropy(self, y):
        # Shannon Entropy - Σ p(c) log2 p(c)
        classes, counts = jnp.unique(y, return_counts=True)
        probs = counts / len(y)
        probs = jnp.where(probs == 0, 1.0, probs)
        return -jnp.sum(probs * jnp.log2(probs))

    def _information_gain(self, y, left_mask, right_mask):
        n = len(y)
        n_left = jnp.sum(left_mask)
        n_right = n - n_left
        
        # Entropy before split
        parent_entropy = self._entropy(y)
        # Entropy on chlid
        child_entropy = (n_left / n) * self._entropy(y[left_mask]) + \
                        (n_right / n) * self._entropy(y[right_mask])
        return parent_entropy - child_entropy

    def best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= self.min_samples_split:
            return None, None 
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            thresholds = jnp.unique(X[:, feature_idx])
            if len(thresholds) <= 1:
                continue
            for threshold in thresholds[:-1]:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if jnp.sum(left_mask) < self.min_samples_leaf or jnp.sum(right_mask) < self.min_samples_leaf:
                    continue
                gain = self._information_gain(y, left_mask, right_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        if best_gain <= 0:
            return None, None
            
        return best_feature, best_threshold

    def make_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(jnp.unique(y)) == 1:
            classes, counts = jnp.unique(y, return_counts=True)
            return Node(None, None, None, None, classes[jnp.argmax(counts)])
        
        # Best split
        feature_idx, threshold = self.best_split(X, y)
        if feature_idx is None:
            classes, counts = jnp.unique(y, return_counts=True)
            return Node(None, None, None, None, classes[jnp.argmax(counts)])
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Make subtrees
        left_subtree = self.make_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.make_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature_idx, threshold, left_subtree, right_subtree, None)

    def fit(self, X, y):
        self.classes = jnp.unique(y)
        self.tree = self.make_tree(X, y)
        return self

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            pred = self.predict_sample(X[i], self.tree)
            predictions.append(pred)
        return jnp.array(predictions)