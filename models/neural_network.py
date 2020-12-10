import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    cross_validate,
    ShuffleSplit,
    validation_curve,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier


from model_analysis import plot_learning_curve
import util


## Constants
RAND_STATE = 42
DATA = "../saved/cleaned_data_Dec9.csv"

FEATURE = ["AuthorRank", "JournalNameRank", "PublisherRank"]
TARGET = "BinnedCitations"


## Data Import
data = pd.read_csv(DATA)

## Data splitting
X, X_test, y, y_test = train_test_split(
    data.loc[:, FEATURE].values,
    data.loc[:, TARGET].values,
    test_size=0.10,
    shuffle=True,
    random_state=RAND_STATE,
)

X_train, X_dev, y_train, y_dev = train_test_split(
    X, y, test_size=0.15, shuffle=True, random_state=RAND_STATE
)

print("Feature size - Train ", X_train.shape)
print("Feature size - Dev ", X_dev.shape)
print("Feature size - Test ", X_test.shape)

## Get uniform version of train and dev set
X_train_uniform, y_train_uniform = util.get_uniform_version(X_train, y_train)
X_dev_uniform, y_dev_uniform = util.get_uniform_version(X_dev, y_dev)

print("Uniform Feature size - Train ", X_train_uniform.shape)
print("Uniform Feature size - Dev ", X_dev_uniform.shape)


## Model Evaluation
ss = StandardScaler()
nn = MLPClassifier(
    hidden_layer_sizes=(8, 16, 16), max_iter=500, alpha=2.0, learning_rate="adaptive"
)
model_pipeline = Pipeline(steps=[("ss", ss), ("nn", nn)])

title = r"Learning Curves (MLP Classifier)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.15, random_state=RAND_STATE)
estimator = Pipeline(steps=[("ss", ss), ("nn", nn)])
fig = plot_learning_curve(
    estimator, title, *util.get_uniform_version(X, y), cv=cv, n_jobs=-1
)
plt.show()


model_pipeline.fit(X_train_uniform, y_train_uniform)
print("Print training score: ", model_pipeline.score(X_train_uniform, y_train_uniform))
ll = model_pipeline.named_steps["nn"].loss_curve_

plt.figure()
plt.title("MLP Classifier Training Loss")
plt.plot(ll)
plt.ylabel("Loss Value")
plt.xlabel("Epochs")
plt.show()
