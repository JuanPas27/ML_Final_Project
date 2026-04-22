import jax.numpy as jnp
from MLP import MLP

class MLPAdapter(MLP):
    def __init__(self, *args, epochs=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = epochs

    def fit(self, X, y):
        y_oh = jnp.eye(self.output_size)[y]
        return self.train_matrix(X, y_oh, epochs=self.epochs, verbose=False)