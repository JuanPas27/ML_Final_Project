import jax.numpy as jnp

class LDA:
    """
    Attempt of Linear Discriminant Analysis implementation as ESL (Chapter 4.3).
    Assuming that all classes share the same covariance matrix.
    """
    
    def __init__(self):
        self.classes = None # Unique class labels
        self.means = None # Class mean vectors μ_k
        self.priors = None # Class-prior probabilities (π_k^ = N_k / N)
        self.cov = None # Shared covariance matrix
        self.cov_inv = None # Inverse of the covariance matrix

    def fit(self, X, y):
        # Unique class labels
        self.classes = jnp.unique(y)
        n_samples, n_features = X.shape
        K = len(self.classes)

        # Estimate class priors π_k
        self.priors = jnp.array([
            jnp.sum(y == c) / n_samples
            for c in self.classes
        ])
        # Estimate class means μ_k
        self.means = jnp.array([
            jnp.mean(X[y == c], axis=0)
            for c in self.classes
        ])
        # Initializing shared covariance matrix
        cov = jnp.zeros((n_features, n_features))
        for i, c in enumerate(self.classes):
            Xc = X[y == c]
            centered = Xc - self.means[i]
            cov += centered.T @ centered
        cov /= (n_samples - K)
        self.cov = cov

        # Inverse covariance matrix Σ^{-1}
        self.cov_inv = jnp.linalg.inv(cov)

        return self

    def delta(self, X, k):
        """
        Discriminant function δ_k(x) for class k.
        """
        mu_k = self.means[k]
        term1 = X @ self.cov_inv @ mu_k
        term2 = - 0.5 * (mu_k @ self.cov_inv @ mu_k)
        term3 = jnp.log(self.priors[k])

        return term1 + term2 + term3

    def decision_rule(self, X):
        """
        Calculate δ_k(x) for each class.
        """
        rules = []
        for k in range(len(self.classes)):
            rules.append(self.delta(X, k))

        return jnp.stack(rules, axis=1)

    def predict(self, X):
        """
        Classify using
            argmax_k δ_k(x)
        """
        scores = self.decision_rule(X)
        idx = jnp.argmax(scores, axis=1)

        return self.classes[idx]