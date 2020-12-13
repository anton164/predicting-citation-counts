import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    cross_val_score,
    ShuffleSplit,
    validation_curve,
)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

from model_analysis import plot_learning_curve
import util


## Constants
RAND_STATE = 42
DATA = "../saved/final_dec11.csv"

META = [
    "JournalNameRankNormalized",
    # "PublisherRankNormalized",
    "AuthorRank",
    # "AuthorProminence",
    "PageCount",
]
TARGET = "BinnedCitationsPerYear"


## Data Import
df = pd.read_csv(DATA)


n_samples = df.shape[0]
test_split = 0.1
dev_split = 0.1


X, X_test, y, y_test = train_test_split(
    df.iloc[:, :-1], df.iloc[:, -1], test_size=0.1, shuffle=True, random_state=42
)

text_features = X.iloc[:, :-6]
meta_features = X.iloc[:, -6:-1]
target = X.iloc[:, -1]

U, V, w, x = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

txt_clsf = MLPClassifier()
txt_clsf.fit(U.iloc[:, :-6], w)

U1 = txt_clsf.predict(U.iloc[:, :-6]).reshape(-1, 1)
V1 = txt_clsf.predict(V.iloc[:, :-6]).reshape(-1, 1)

print("Txt classifier: ")
print(classification_report(x, V1))

meta_clsf = RandomForestClassifier()
meta_clsf.fit(U[:, -6:-1], w)

U2 = meta_clsf.predict(U[:, -6:-1]).reshape(-1, 1)
V2 = meta_clsf.predict(V[:, -6:-1]).reshape(-1, 1)

print("Meta classifier: ")
print(classification_report(x, V2))


ensemble_clsf = LinearSVC()
ensemble_clsf.fit(np.concatenate((U1, U2), axis=1), w)

print("Ensemble classifier: ")
print(classification_report(x, ensemble_clsf.predict(np.concatenate((V1, V2), axis=1))))
