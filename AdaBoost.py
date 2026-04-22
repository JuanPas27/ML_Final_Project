import jax.numpy as jnp
import numpy as np
import copy

class AdaBoost:
    def __init__(self, base_estimator, n_estimators=50, learning_rate=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        self.classes = None

    def _calculate_error(self, y_true, y_pred, sample_weights):
        incorrect = (y_pred != y_true)
        return jnp.sum(sample_weights * incorrect) / jnp.sum(sample_weights)

    def fit(self, X, y):
        self.classes = jnp.unique(y)
        n_samples = X.shape[0]

        # 1. Initialize the observation weights wi = 1/N, i = 1,2,...,N
        sample_weights = jnp.ones(n_samples) / n_samples

        # 2. For m = 1 to M:
        for _ in range(self.n_estimators):
            # (a) Fit a classifier Gm(x) to the training data using weights wi
            estimator = copy.deepcopy(self.base_estimator)
            rng = np.random.RandomState()
            indices = rng.choice(n_samples, size=n_samples, p=np.array(sample_weights))
            X_weighted = X[indices]
            y_weighted = y[indices]
            estimator.fit(X_weighted, y_weighted)
            y_pred = estimator.predict(X)

            # (b) Compute err_m
            error = self._calculate_error(y, y_pred, sample_weights)
            if error >= 0.5 or error == 0:
                if error == 0:
                    self.estimators.append(estimator)
                    self.estimator_weights.append(1.0)
                continue

            # (c) Compute αm = log((1−errm)/errm)
            alpha = self.learning_rate * jnp.log((1 - error) / error)

            # (d) Set wi ← wi ·exp[αm ·I(yi= Gm(xi))], i = 1,2,...,N
            sample_weights = sample_weights * jnp.exp(alpha * (y_pred != y))
            sample_weights = sample_weights / jnp.sum(sample_weights)

            # Save
            self.estimators.append(estimator)
            self.estimator_weights.append(alpha)

            if error < 1e-6:
                break

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        scores = jnp.zeros((n_samples, n_classes))
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            y_pred = estimator.predict(X)
            for i, pred in enumerate(y_pred):
                class_idx = jnp.where(self.classes == pred)[0][0]
                scores = scores.at[i, class_idx].add(weight)

        return self.classes[jnp.argmax(scores, axis=1)]