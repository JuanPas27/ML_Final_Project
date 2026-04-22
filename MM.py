import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

class MixtureModels:
    def __init__(self, n_components_per_class=3, max_iter=100, tol=1e-3, reg_covar=1e-6, seed=42):
        self.n_components = n_components_per_class
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.key = jax.random.PRNGKey(seed)
        self.classes = None
        self.class_models = {}  # {class_label: (weights, means, covariances, prior)}

    def initialize_parameters(self, X):
        n_samples, n_features = X.shape
        weights = jnp.ones(self.n_components) / self.n_components
        # Means
        idx = jax.random.choice(self.key, n_samples, (self.n_components,), replace=False)
        means = X[idx]
        # Covariance matrix
        cov_init = jnp.cov(X.T) + self.reg_covar * jnp.eye(n_features)
        covariances = jnp.stack([cov_init] * self.n_components)
        return weights, means, covariances

    def e_step(self, X, weights, means, covariances):
        n_components = len(weights)
        log_probs = []
        for k in range(n_components):
            diff = X - means[k]
            inv_cov = jnp.linalg.inv(covariances[k])
            log_det = jnp.linalg.slogdet(covariances[k])[1]
            log_p = -0.5 * (jnp.sum(diff @ inv_cov * diff, axis=1) + log_det + X.shape[1] * jnp.log(2 * jnp.pi))
            log_probs.append(log_p + jnp.log(weights[k]))
        log_probs = jnp.stack(log_probs, axis=1)
        log_resp = log_probs - logsumexp(log_probs, axis=1, keepdims=True)
        
        return jnp.exp(log_resp)

    def m_step(self, X, resp):
        n_components = resp.shape[1]
        n_samples, n_features = X.shape
        nk = jnp.sum(resp, axis=0) + 1e-10
        weights = nk / n_samples
        means = jnp.dot(resp.T, X) / nk[:, None]
        covariances = []
        for k in range(n_components):
            diff = X - means[k]
            cov_k = jnp.dot(resp[:, k] * diff.T, diff) / nk[k]
            cov_k += self.reg_covar * jnp.eye(n_features)
            covariances.append(cov_k)

        return weights, means, jnp.stack(covariances)

    def fit_one_class(self, X):
        weights, means, covariances = self.initialize_parameters(X)
        log_likelihood_old = -jnp.inf
        for _ in range(self.max_iter):
            # E step
            resp = self.e_step(X, weights, means, covariances)
            # M step
            weights, means, covariances = self.m_step(X, resp)
            log_likelihood = jnp.sum(logsumexp(jnp.log(weights) + self.e_step(X, weights, means, covariances), axis=1))
            if jnp.abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood

        return weights, means, covariances

    def fit(self, X, y):
        self.classes = jnp.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            prior = len(X_c) / len(X)
            weights, means, covs = self.fit_one_class(X_c)
            self.class_models[int(c)] = (weights, means, covs, prior)
        return self

    def log_density(self, X, c):
        weights, means, covariances, prior = self.class_models[int(c)]
        log_densities = []
        for k in range(self.n_components):
            diff = X - means[k]
            inv_cov = jnp.linalg.inv(covariances[k])
            log_det = jnp.linalg.slogdet(covariances[k])[1]
            log_p = -0.5 * (jnp.sum(diff @ inv_cov * diff, axis=1) + log_det + X.shape[1] * jnp.log(2 * jnp.pi))
            log_densities.append(log_p)
        log_densities = jnp.stack(log_densities, axis=1)
        log_sum = logsumexp(log_densities + jnp.log(weights), axis=1)
        return log_sum + jnp.log(prior)

    def predict_classes(self, X):
        log_probs = []
        for c in self.classes:
            log_probs.append(self.log_density(X, c))
        log_probs = jnp.stack(log_probs, axis=1)
        return jnp.exp(log_probs - logsumexp(log_probs, axis=1, keepdims=True))

    def predict(self, X):
        probs = self.predict_classes(X)
        return self.classes[jnp.argmax(probs, axis=1)]