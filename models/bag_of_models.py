from typing import Any, Dict
import numpy as np
import json
import os


# 5 types of models
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# model evaluation functions
from sklearn.metrics import f1_score


class BagOfModels:
    def __init__(self) -> None:
        self.models_ = {
            "LinearSVC": SVC,
            "NaiveBayes": MultinomialNB,
            "RandomForest": RandomForestClassifier,
            "NeuralNetwork": MLPClassifier,
            "XGBoost": XGBClassifier,
        }
        self.fit_models_ = {}
        self.training_scores_ = {}
        self.score_fn = f1_score
        self.hyperparams = {}

    def fit(
        self, X: np.ndarray, y: np.ndarray, hyperparams: Dict[str, dict] = None
    ) -> None:
        """
        Initializes model with specified hyperparameters, and
        trains model with provided training data.

        Collects training scores during training.

        Args:
            X: np.ndarray - Feature data
            y: np.ndarray - Ground truth classes
            hyperparams: dict - Dict of dicts with hyperparameter options for model init.

        Returns:
            None
        """
        if hyperparams is None:
            hyperparams = self.hyperparams
        else:
            self.hyperparams = hyperparams

        for model_name, model_fn in self.models_.items():
            # init model with hyperparameter selection
            params = hyperparams.get(model_name, {})

            # option to skip a selected model
            if params is False:
                continue
            print("Fitting model: ", model_name)

            model = model_fn(**params)

            # train model and save trained model to dict
            self.fit_models_[model_name] = model
            model.fit(X, y)

            # calculate training scores
            y_pred = model.predict(X)
            self.training_scores_ = self.score(X, y_pred)

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Runs prediction on all trained models.

        Args:
            X: np.ndarray - Feature data
            y: np.ndarray - Ground truth classes

        Returns:
            Dict[str, np.ndarray]: Predictions for of each model
        """
        labels = {}
        for model_name, model in self.fit_models_.items():
            y_pred = model.predict(X)
            labels[model_name] = y_pred

        return labels

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Returns prediction scores for all trained models.

        Args:
            X: np.ndarray - Feature data
            y: np.ndarray - Ground truth classes

        Returns:
            Dict[str, float]: Scores of each model
        """
        scores = {}
        for model_name, model in self.fit_models_.items():
            y_pred = model.predict(X)
            scores[model_name] = self.score_fn(y, y_pred)

        return scores

    def load_hyperparams(self, filepath: str) -> None:
        """
        Loads JSON file with hyperparameter options for models.

        Args:
            filepath: string - Path to hyperparameter JSON file.

        Returns:
            None
        """
        with open(filepath, "r") as file:
            self.hyperparams = json.load(file)

    def dump_hyperparams(self, filepath: str) -> None:
        """
        Writes JSON file with hyperparameter options for models.

        Args:
            filepath: string - Path to hyperparameter JSON file.

        Returns:
            None
        """
        with open(filepath, "w") as file:
            json.dump(self.hyperparams, file, sort_keys=True, indent=4)

    def get_best_estimator(self) -> Any:
        """
        Returns estimator that achieved the highest prediction score.

        Args:
            None

        Returns:
            Estimator instance
        """
        best_score = -np.infty
        best_model = None
        for model_name, _ in self.fit_models_.items():
            score = self.validation_scores_.get(model_name, -1.0)
            if score > best_score:
                best_score = score
                best_model = model_name

        if best_score < 0.0:
            print(
                "WARNING: No validation scores available. \n \t -> Run BagOfModels.predict(X, y) first."
            )

        return self.fit_models_[best_model]


if __name__ == "__main__":
    bom = BagOfModels()

    hyperparams = {
        "SVC": {"kernel": "linear"},
        "NeuralNetwork": {"early_stopping": True},
    }
    X = np.eye(100, 5, dtype=float)
    y = np.ones(100, dtype=int)
    y[:50] = 0

    bom.fit(X, y, hyperparams=hyperparams)
    print(bom.training_scores_)

    bom.dump_hyperparams("./test.json")
    bom.load_hyperparams("./test.json")

    y_pred = bom.predict(X, y)
    print(y_pred)
    print(bom.validation_scores_)
