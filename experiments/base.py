import numpy as np
import pickle
from sklearn.model_selection import (
    train_test_split,
)


class Experiment:
    def __init__(self, pandas_df):
        self.X = pandas_df.iloc[:, :-1]
        self.y = pandas_df.iloc[:, -1]
        self.model = None

    def run(self):
        """
        Called by streamlit interface.
        Should hold all functions you want to execute to preprocess and train your model.
        """
        pass

    def split(self, split=0.8):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=split, random_state=42
        )

    def preprocess(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def save(self, filename):
        if self.model is not None:
            pickle.dump(self.model, open(filename, "wb"))

    def load(self, filename):
        if self.model is None:
            self.model = pickle.load(open(filename, "rb"))
        else:
            return "Error: Can't overwrite existing experiment"
