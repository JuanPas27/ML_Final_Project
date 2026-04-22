import os
import sys
# Importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
import pickle
import mlflow
mlflow.set_tracking_uri("file:./mlruns") # Help with paralelism of models
from metaflow import FlowSpec, step, Parameter

# Import classes
from LDA import LDA
from Logistic_Regression import LogisticRegression
from MLP import MLP
from MM import MixtureModels
from DecisionTree import DecisionTree
from metrics import precision_recall_f1

class MusicGenreFlow(FlowSpec):
    # Params
    test_size = Parameter('test_size', help='Size of the test data', default=0.2)
    random_seed = Parameter('random_seed', help='Random seed', default=42)
    max_iter = Parameter('max_iter', help='Iterations for logistic regression', default=100)
    tol = Parameter('tol', help='Epsilon for logistic regression', default=1e-6)

    @step
    def start(self):
        """Load Data from GTZAN dataset CSV"""
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'data', 'features_3_sec.csv')
        self.raw_data = pd.read_csv(data_path)
        print(f"Data: {self.raw_data.shape}")
        os.makedirs("models", exist_ok=True)
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """Preprocessing data"""
        # Map labels
        genres = self.raw_data['label'].unique()
        genre_to_id = {g: i for i, g in enumerate(genres)}
        self.raw_data['label_id'] = self.raw_data['label'].map(genre_to_id)
        self.genre_names = genres

        # Features and target
        X = self.raw_data.drop(['filename', 'length', 'label', 'label_id'], axis=1)
        y = self.raw_data['label_id'].values

        # Convert to jax arrays
        X_jax = jnp.array(X.values)
        y_jax = jnp.array(y)

        # Split dataset with random key
        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(X_jax))
        test_len = int(len(X_jax) * self.test_size)
        test_idx = indices[:test_len]
        train_idx = indices[test_len:]

        X_train = X_jax[train_idx]
        X_test = X_jax[test_idx]
        y_train = y_jax[train_idx]
        y_test = y_jax[test_idx]

        # Standarize data
        mean = jnp.mean(X_train, axis=0)
        std = jnp.std(X_train, axis=0)
        std = jnp.where(std == 0, 1.0, std)
        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std

        # Save in global
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test

        # One-hot encoding (0/1) for softmax and cross-entropy application (MLP)
        def one_hot(y, num_classes=10):
            return jnp.eye(num_classes)[y]
        self.y_train_onehot = one_hot(self.y_train)
        self.y_test_onehot  = one_hot(self.y_test)

        # Augment features with intercept
        ones_train = jnp.ones((X_train_scaled.shape[0], 1))
        ones_test = jnp.ones((X_test_scaled.shape[0], 1))
        self.X_train_aug = jnp.concatenate([ones_train, X_train_scaled], axis=1)
        self.X_test_aug = jnp.concatenate([ones_test, X_test_scaled], axis=1)

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        self.next(self.train_lda, self.train_logistic, self.train_mlp,
                  self.train_mm, self.train_decision_tree)

    @step
    def train_lda(self):
        """Train LDA model"""
        with mlflow.start_run(run_name="LDA_GTZAN"):
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("random_seed", self.random_seed)
            mlflow.log_param("model_type", "LDA")

            model = LDA()
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)

            precisions, recalls, f1s = precision_recall_f1(self.y_test, y_pred)
            precision = float(jnp.mean(jnp.array(precisions)))
            recall = float(jnp.mean(jnp.array(recalls)))
            f1 = float(jnp.mean(jnp.array(f1s)))

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            # Save artifact model
            model_path = "models/lda_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)

            self.lda_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
            print(f"LDA -> F1: {f1:.4f}")

        self.next(self.join)

    @step
    def train_logistic(self):
        """Train Logistic Regression model"""
        with mlflow.start_run(run_name="Logistic_GTZAN"):
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("random_seed", self.random_seed)
            mlflow.log_param("max_iter", self.max_iter)
            mlflow.log_param("tol", self.tol)
            mlflow.log_param("model_type", "LogisticRegression")

            model = LogisticRegression(max_iter=self.max_iter, tol=self.tol)
            model.fit(self.X_train_aug, self.y_train)
            y_pred = model.predict(self.X_test_aug)

            precisions, recalls, f1s = precision_recall_f1(self.y_test, y_pred)
            precision = float(jnp.mean(jnp.array(precisions)))
            recall = float(jnp.mean(jnp.array(recalls)))
            f1 = float(jnp.mean(jnp.array(f1s)))

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            # Save artifact model
            model_path = "models/logistic_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)

            self.logistic_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
            print(f"Logistic -> F1: {f1:.4f}")

        self.next(self.join)

    @step
    def train_mlp(self):
        """Train Multilayer Perceptron"""
        with mlflow.start_run(run_name="MLP_GTZAN"):
            # Hiperparams
            hidden_size = 100
            learning_rate = 0.5
            epochs = 100
            mlflow.log_param("hidden_size", hidden_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("model_type", "MLP")

            input_size = self.X_train_scaled.shape[1]
            output_size = len(self.genre_names)

            model = MLP(input_size, hidden_size, output_size, learning_rate=learning_rate)
            model.train_matrix(self.X_train_scaled, self.y_train_onehot,
                            x_val=self.X_test_scaled, y_val=self.y_test_onehot,
                            epochs=epochs, verbose=True)

            y_pred = model.predict(self.X_test_scaled)
            precisions, recalls, f1s = precision_recall_f1(self.y_test, y_pred)
            precision = float(jnp.mean(jnp.array(precisions)))
            recall = float(jnp.mean(jnp.array(recalls)))
            f1 = float(jnp.mean(jnp.array(f1s)))

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            # Save artifact model
            model_path = "models/mlp_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)

            self.mlp_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
            print(f"MLP -> F1: {f1:.4f}")

        self.next(self.join)

    @step
    def train_mm(self):
        """Train Mixture Models"""
        with mlflow.start_run(run_name="GMM_GTZAN"):
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("random_seed", self.random_seed)
            mlflow.log_param("model_type", "GMM")
            mlflow.log_param("n_components", 3)

            model = MixtureModels(n_components_per_class=3, max_iter=50)
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)

            precisions, recalls, f1s = precision_recall_f1(self.y_test, y_pred)
            precision = float(jnp.mean(jnp.array(precisions)))
            recall = float(jnp.mean(jnp.array(recalls)))
            f1 = float(jnp.mean(jnp.array(f1s)))

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            # Save artifact model
            model_path = "models/mm_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)

            self.mm_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
            print(f"MM -> F1: {f1:.4f}")

        self.next(self.join)

    @step
    def train_decision_tree(self):
        """Train Decision Tree"""
        with mlflow.start_run(run_name="DecisionTree_GTZAN"):
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("random_seed", self.random_seed)
            mlflow.log_param("model_type", "DecisionTree")
            mlflow.log_param("max_depth", 10)

            model = DecisionTree(max_depth=10, min_samples_split=5)
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)

            precisions, recalls, f1s = precision_recall_f1(self.y_test, y_pred)
            precision = float(jnp.mean(jnp.array(precisions)))
            recall = float(jnp.mean(jnp.array(recalls)))
            f1 = float(jnp.mean(jnp.array(f1s)))

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            # Save artifact model
            model_path = "models/decision_tree_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)

            self.dt_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
            print(f"Decision Tree -> F1: {f1:.4f}")

        self.next(self.join)

    @step
    def join(self, inputs):
        """Comparing results"""
        lda = inputs.train_lda.lda_metrics
        log = inputs.train_logistic.logistic_metrics
        mlp = inputs.train_mlp.mlp_metrics
        mm = inputs.train_mm.mm_metrics
        dt = inputs.train_decision_tree.dt_metrics

        all_metrics = {
            'LDA': lda,
            'Logistic': log,
            'MLP': mlp,
            'MM': mm,
            'DecisionTree': dt
        }

        print("\n" + "="*50)
        print("Comparing models")
        print("="*50)
        print(f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-"*50)
        for name, m in all_metrics.items():
            print(f"{name:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}")
        print("="*50)

        best_f1 = max(all_metrics.items(), key=lambda x: x[1]['f1'])
        print(f"Best model: {best_f1[0]} with F1 = {best_f1[1]['f1']:.4f}")

        self.next(self.end)

    @step
    def end(self):
        print("\nPipeline completed successful.")

if __name__ == '__main__':
    MusicGenreFlow()