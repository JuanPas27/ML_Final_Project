import os
import sys
# Importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import jax.numpy as jnp
import pickle
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
from metaflow import FlowSpec, step, Parameter
import json

# Import classes
from LDA import LDA
from Logistic_Regression import LogisticRegression
from MLP import MLP
from MLPAdapter import MLPAdapter
from MM import MixtureModels
from DecisionTree import DecisionTree
from AdaBoost import AdaBoost
from metrics import precision_recall_f1, confusion_matrix

class AdaBoostMusicGenreFlow(FlowSpec):
    # Params
    test_size = Parameter('test_size', help='Size of the test data', default=0.2)
    random_seed = Parameter('random_seed', help='Random seed', default=42)
    n_estimators = Parameter('n_estimators', help='Number of estimators', default=10)
    learning_rate = Parameter('learning_rate', help='Learning rate', default=1.0)

    @step
    def start(self):
        """Load Data from GTZAN dataset CSV"""
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'data', 'features_3_sec.csv')
        self.raw_data = pd.read_csv(data_path)
        print(f"Data: {self.raw_data.shape}")
        os.makedirs("models_boost", exist_ok=True)
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """Preprocessing data"""
        # Map labels
        genres = self.raw_data['label'].unique()
        genre_to_id = {g: i for i, g in enumerate(genres)}
        self.raw_data['label_id'] = self.raw_data['label'].map(genre_to_id)
        self.genre_names = genres

        with open("models_boost/genre_names.pkl", "wb") as f:
            pickle.dump(self.genre_names, f)

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
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test

        # One-hot encoding (0/1) for softmax and cross-entropy application (MLP)
        def one_hot(y, num_classes=10):
            return jnp.eye(num_classes)[y]
        self.y_train_onehot = one_hot(self.y_train)
        self.y_test_onehot = one_hot(self.y_test)

        # Augment features with intercept
        ones_train = jnp.ones((X_train_scaled.shape[0], 1))
        ones_test = jnp.ones((X_test_scaled.shape[0], 1))
        self.X_train_aug = jnp.concatenate([ones_train, X_train_scaled], axis=1)
        self.X_test_aug = jnp.concatenate([ones_test, X_test_scaled], axis=1)

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        self.next(self.train_lda, self.train_logistic, self.train_mlp, 
                  self.train_mm, self.train_dt)
    
    @step
    def train_lda(self):
        """Train LDA model"""
        with mlflow.start_run(run_name="AdaBoost_LDA"):
            mlflow.log_params({"test_size": self.test_size, "n_estimators": self.n_estimators})
            base = LDA()
            model = AdaBoost(base_estimator=base, n_estimators=self.n_estimators, learning_rate=self.learning_rate)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            cm = confusion_matrix(self.y_test, y_pred, num_classes=len(self.genre_names))
            cm_path = f"models_boost/confusion_LDA_boost.json"
            with open(cm_path, "w") as f:
                json.dump(cm.tolist(), f)
            mlflow.log_artifact(cm_path)

            precisions, recalls, f1s = precision_recall_f1(self.y_test, y_pred)
            precision = float(jnp.mean(jnp.array(precisions)))
            recall = float(jnp.mean(jnp.array(recalls)))
            f1 = float(jnp.mean(jnp.array(f1s)))
            mlflow.log_metrics({'precision': precision, 'recall': recall, 'f1': f1})

            # Save artifact model
            path = "models_boost/lda_boost.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(path)

            self.lda_boost_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
            print(f"AdaBoost + LDA -> F1: {f1:.4f}")
        
        self.next(self.join)
    
    @step
    def train_logistic(self):
        with mlflow.start_run(run_name="AdaBoost_Logistic"):
            mlflow.log_params({"test_size": self.test_size, "n_estimators": self.n_estimators})
            base = LogisticRegression(max_iter=100, tol=1e-6)
            model = AdaBoost(base_estimator=base, n_estimators=self.n_estimators, learning_rate=self.learning_rate)
            model.fit(self.X_train_aug, self.y_train)
            y_pred = model.predict(self.X_test_aug)

            cm = confusion_matrix(self.y_test, y_pred, num_classes=len(self.genre_names))
            cm_path = f"models_boost/confusion_LOG_boost.json"
            with open(cm_path, "w") as f:
                json.dump(cm.tolist(), f)
            mlflow.log_artifact(cm_path)

            precisions, recalls, f1s = precision_recall_f1(self.y_test, y_pred)
            precision = float(jnp.mean(jnp.array(precisions)))
            recall = float(jnp.mean(jnp.array(recalls)))
            f1 = float(jnp.mean(jnp.array(f1s)))
            mlflow.log_metrics({'precision': precision, 'recall': recall, 'f1': f1})

            # Save artifact model
            path = "models_boost/logistic_boost.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(path)

            self.logistic_boost_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
            print(f"AdaBoost + Logistic -> F1: {f1:.4f}")
        self.next(self.join)
    
    @step
    def train_mlp(self):
        with mlflow.start_run(run_name="AdaBoost_MLP"):
            mlflow.log_params({"test_size": self.test_size, "n_estimators": self.n_estimators})
            input_size = self.X_train.shape[1]
            output_size = len(self.genre_names)
            base = MLPAdapter(input_size, hidden_size=100, output_size=output_size, learning_rate=0.5)
            model = AdaBoost(base_estimator=base, n_estimators=self.n_estimators, learning_rate=self.learning_rate)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            cm = confusion_matrix(self.y_test, y_pred, num_classes=len(self.genre_names))
            cm_path = f"models_boost/confusion_MLP_boost.json"
            with open(cm_path, "w") as f:
                json.dump(cm.tolist(), f)
            mlflow.log_artifact(cm_path)

            precisions, recalls, f1s = precision_recall_f1(self.y_test, y_pred)
            precision = float(jnp.mean(jnp.array(precisions)))
            recall = float(jnp.mean(jnp.array(recalls)))
            f1 = float(jnp.mean(jnp.array(f1s)))
            mlflow.log_metrics({'precision': precision, 'recall': recall, 'f1': f1})
            
            # Save artifact model
            path = "models_boost/mlp_boost.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(path)

            self.mlp_boost_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
            print(f"AdaBoost + MLP -> F1: {f1:.4f}")
        self.next(self.join)
    
    @step
    def train_mm(self):
        with mlflow.start_run(run_name="AdaBoost_MM"):
            mlflow.log_params({"test_size": self.test_size, "n_estimators": self.n_estimators})
            base = MixtureModels(n_components_per_class=3, max_iter=50)
            model = AdaBoost(base_estimator=base, n_estimators=self.n_estimators, learning_rate=self.learning_rate)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            cm = confusion_matrix(self.y_test, y_pred, num_classes=len(self.genre_names))
            cm_path = f"models_boost/confusion_MM_boost.json"
            with open(cm_path, "w") as f:
                json.dump(cm.tolist(), f)
            mlflow.log_artifact(cm_path)

            precisions, recalls, f1s = precision_recall_f1(self.y_test, y_pred)
            precision = float(jnp.mean(jnp.array(precisions)))
            recall = float(jnp.mean(jnp.array(recalls)))
            f1 = float(jnp.mean(jnp.array(f1s)))
            mlflow.log_metrics({'precision': precision, 'recall': recall, 'f1': f1})

            # Save artifact model
            path = "models_boost/mm_boost.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(path)

            self.mm_boost_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
            print(f"AdaBoost + MM -> F1: {f1:.4f}")
        self.next(self.join)
    
    @step
    def train_dt(self):
        with mlflow.start_run(run_name="AdaBoost_DT"):
            mlflow.log_params({"test_size": self.test_size, "n_estimators": self.n_estimators})
            base = DecisionTree(max_depth=10, min_samples_split=5)
            model = AdaBoost(base_estimator=base, n_estimators=self.n_estimators, learning_rate=self.learning_rate)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            cm = confusion_matrix(self.y_test, y_pred, num_classes=len(self.genre_names))
            cm_path = f"models_boost/confusion_DT_boost.json"
            with open(cm_path, "w") as f:
                json.dump(cm.tolist(), f)
            mlflow.log_artifact(cm_path)

            precisions, recalls, f1s = precision_recall_f1(self.y_test, y_pred)
            precision = float(jnp.mean(jnp.array(precisions)))
            recall = float(jnp.mean(jnp.array(recalls)))
            f1 = float(jnp.mean(jnp.array(f1s)))
            mlflow.log_metrics({'precision': precision, 'recall': recall, 'f1': f1})

            # Save artifact model
            path = "models_boost/dt_boost.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(path)

            self.dt_boost_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
            print(f"AdaBoost + DT -> F1: {f1:.4f}")
        self.next(self.join)
    
    @step
    def join(self, inputs):
        metrics = {
            "LDA + AdaBoost": inputs.train_lda.lda_boost_metrics,
            "Logistic + AdaBoost": inputs.train_logistic.logistic_boost_metrics,
            "MLP + AdaBoost": inputs.train_mlp.mlp_boost_metrics,
            "MM + AdaBoost": inputs.train_mm.mm_boost_metrics,
            "DT + AdaBoost": inputs.train_dt.dt_boost_metrics,
        }

        print("\n" + "="*60)
        print("Comparing models boosted")
        print("="*60)
        print(f"{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-"*60)
        for name, m in metrics.items():
            print(f"{name:<25} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}")
        print("="*60)

        best = max(metrics.items(), key=lambda x: x[1]['f1'])
        print(f"Best boosted model: {best[0]} with F1 = {best[1]['f1']:.4f}")

        metrics_path = "models_boost/metrics_boost.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Boosted metrics saved to {metrics_path}")

        self.next(self.end)

    @step
    def end(self):
        print("\nAdaBoost pipeline completed successfully.")

if __name__ == '__main__':
    AdaBoostMusicGenreFlow()