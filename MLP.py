import jax
import jax.numpy as jnp
from jax import random, grad, jit

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5, sigmoid_alpha=1.0, big_real=30.0, seed=0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.sigmoid_alpha = sigmoid_alpha
        self.big_real = big_real
        self.key = random.PRNGKey(seed)
        self.params = self.init_weights()

    def init_weights(self):
        key1, key2 = random.split(self.key)
        # Input-to-hidden
        scale1 = jnp.sqrt(1.0 / (self.input_size + 1))
        W1 = random.normal(key1, (self.input_size + 1, self.hidden_size)) * scale1
        # Hidden-to-output
        scale2 = jnp.sqrt(1.0 / (self.hidden_size + 1))
        W2 = random.normal(key2, (self.hidden_size + 1, self.output_size)) * scale2
        return (W1, W2)

    def sigmoid(self, v):
        v_clipped = jnp.clip(v, -self.big_real, self.big_real)
        return 1.0 / (1.0 + jnp.exp(-self.sigmoid_alpha * v_clipped))

    def sigmoid_derivate(self, v):
        fv = self.sigmoid(v)
        return self.sigmoid_alpha * fv * (1 - fv)

    def softmax(self, x):
        x_max = jnp.max(x, axis=-1, keepdims=True)
        return jnp.exp(x - x_max) / jnp.sum(jnp.exp(x - x_max), axis=-1, keepdims=True)

    def cross_entropy(self, logits, y_true):
        y_pred = self.softmax(logits)
        return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred + 1e-8), axis=-1))

    def forward(self, x):
        W1, W2 = self.params
        x_aug = jnp.column_stack([x, jnp.ones((x.shape[0], 1))])
        net_h = jnp.dot(x_aug, W1)
        y_h = self.sigmoid(net_h)
        y_h_aug = jnp.column_stack([y_h, jnp.ones((y_h.shape[0], 1))])
        logits = jnp.dot(y_h_aug, W2)
        return logits, y_h, net_h

    def loss(self, x, y):
        logits, _, _ = self.forward(x)
        return self.cross_entropy(logits, y)

    def backpropagation_matrix(self, x, y):
        W1, W2 = self.params
        x_aug = jnp.column_stack([x, jnp.ones((x.shape[0], 1))])
        # Forward pass
        logits, y_h, net_h = self.forward(x)
        probs = self.softmax(logits)
        y_h_aug = jnp.column_stack([y_h, jnp.ones((y_h.shape[0], 1))])
        # Output delta
        delta_o = probs - y
        # Gradients for W2
        dW2 = jnp.dot(y_h_aug.T, delta_o) / x.shape[0]
        # Hidden delta
        delta_h = jnp.dot(delta_o, W2[:-1, :].T) * self.sigmoid_derivate(net_h)
        # Gradients for W1
        dW1 = jnp.dot(x_aug.T, delta_h) / x.shape[0]
        return (dW1, dW2)

    def update_matrix(self, grads):
        self.params = [(p - self.lr * g) for p, g in zip(self.params, grads)]

    def train_matrix(self, x_train, y_train, x_val=None, y_val=None, epochs=100, verbose=True):
        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            grads = self.backpropagation_matrix(x_train, y_train)
            self.update_matrix(grads)
            train_loss = self.loss(x_train, y_train)
            history['train_loss'].append(train_loss)
            if x_val is not None:
                val_loss = self.loss(x_val, y_val)
                history['val_loss'].append(val_loss)
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1:2d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
            else:
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1:2d} | Train loss: {train_loss:.4f}")
        return history

    def train_auto_diff(self, x_train, y_train, x_val=None, y_val=None, epochs=100, verbose=True):
        def loss_fn(params, x, y):
            old_params = self.params
            self.params = params
            loss_val = self.loss(x, y)
            self.params = old_params
            return loss_val
        grad_loss = jit(grad(loss_fn, argnums=0))
        history = {'train_loss': [], 'val_loss': []}
        params = self.params
        for epoch in range(epochs):
            grads = grad_loss(params, x_train, y_train)
            params = [(p - self.lr * g) for p, g in zip(params, grads)]
            self.params = params
            train_loss = loss_fn(params, x_train, y_train)
            history['train_loss'].append(train_loss)
            if x_val is not None:
                val_loss = loss_fn(params, x_val, y_val)
                history['val_loss'].append(val_loss)
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1:2d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
            else:
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1:2d} | Train loss: {train_loss:.4f}")
        return history

    def predict(self, x):
        logits, _, _ = self.forward(x)
        return jnp.argmax(self.softmax(logits), axis=1)