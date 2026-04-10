import jax.numpy as jnp
import jax

class LogisticRegression:
    """
    Attempt of Multiclass Logistic Regression implementation as ESL (Chapter 4.4).
    Using K-1 parameter vectors with the Kth class as reference.
    To handle issues when the Hessian matrix is singular, we use the pseudo-inverse.
    """

    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter # Maximum number of iterations
        self.tol = tol # convergence tolerance on parameter change
        self.theta = None # parameter set of vectors k-1
        self.classes = None # class labels
        self.n_features = None # number of features
        self.K = None # number of classes

    def to_list(self, theta):
        """
        Convert theta into a list of K-1 coefficient vectors.
        """
        betas = []
        start = 0
        for _ in range(self.K - 1):
            betas.append(theta[start:start + self.n_features])
            start += self.n_features
        
        return betas

    def probability(self, X, theta):
        """
        For classes 1,...,K-1:
            p_k(x) = exp(eta_k) / (1 + sum_{l=1}^{K-1} exp(eta_l))
        For the class K:
            p_K(x) = 1 / (1 + sum_{l=1}^{K-1} exp(eta_l))
        """
        betas = self.to_list(theta) # list of K-1 vectors
        # eta_k = X @ beta_k  for k = 1,...,K-1
        eta = jnp.array([jnp.dot(X, beta) for beta in betas]).T  # (n_samples, K-1)
        exp_eta = jnp.exp(eta) # exp(eta_k)

        denom = 1 + jnp.sum(exp_eta, axis=1, keepdims=True)   # denominator

        probs_no_k = exp_eta / denom # probabilities for classes 1..K-1
        prob_k = 1 / denom # probability for class K
        
        return jnp.concatenate([probs_no_k, prob_k], axis=1)

    def loss(self, theta, X, y):
        """
            l(θ) = Σ_i log p_{g_i}(x_i;θ)
        """
        probs = self.probability(X, theta)
        n = X.shape[0]
        real = probs[jnp.arange(n), y] # For each x_i, pick the probability corresponding to the class y[i]

        return -jnp.mean(jnp.log(real))

    def fit(self, X, y):
        self.classes = jnp.unique(y)
        self.K = len(self.classes)
        self.n_features = X.shape[1]

        # Initialise theta = 0
        theta = jnp.zeros((self.K - 1) * self.n_features)
        # Optimize operations with jax
        loss_op = jax.jit(self.loss)
        grad_op = jax.jit(jax.grad(self.loss))
        hessian_op = jax.jit(jax.hessian(self.loss))
        for i in range(self.max_iter):
            loss_old = loss_op(theta, X, y)
            grad = grad_op(theta, X, y)
            Hessian = hessian_op(theta, X, y)

            # Solve H * delta = g
            # assuming Hessian is invertible
            delta = jnp.linalg.solve(Hessian, grad) # Newton update
            theta_new = theta - delta # (quasi Newton methond)

            # Step‑halving: if the loss doesnot decrease, we reduce the step by half
            step_factor = 1.0
            while step_factor > 1e-10:
                theta_candidate = theta - step_factor * delta
                loss_new = loss_op(theta_candidate, X, y)
                if loss_new < loss_old:
                    theta_new = theta_candidate
                    break
                step_factor *= 0.5

            # Check we reach tolerance convergence
            if jnp.linalg.norm(theta_new - theta) < self.tol:
                theta = theta_new
                break
            theta = theta_new

            if i % 10 == 0: # step of iterations report
                print(f"Iteration {i}, loss: {loss_old:.6f}")
        self.theta = theta

        return self

    def predict_proba(self, X):
        """
        Return class probabilities for samples in X
        """
        return jnp.array(self.probability(X, self.theta))

    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        probs = self.predict_proba(X)
        return jnp.argmax(probs, axis=1)