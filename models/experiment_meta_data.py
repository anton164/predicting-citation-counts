import numpy as np
import pandas as pd 

# models
from bag_of_models import BagOfModels

# data prep
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import util
import df_utils

# model evaluation
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

features = df_utils.load_df("../saved/metadata_features.csv")
target = df_utils.load_df("../saved/binned_citations_threshold_1.csv")

X_train, X_test, y_train, y_test = train_test_split(features.iloc[:].values, target.iloc[:].values.ravel(), test_size=0.1, random_state=0)

X, y = util.get_uniform_version(X_train, y_train)

print(X.shape)

hyperparams = {
    'SVC': {
        'kernel': 'linear',
        'random_state': 0
    },
    'NeuralNetwork': {
        'early_stopping': True,
        'random_state': 0
    }
}

bom = BagOfModels()
bom.fit(X, y, hyperparams)

print(bom.training_scores_)

bom.dump_hyperparams('./meta_1.json')

bom.predict(X_test, y_test)

print(bom.validation_scores_)

clsf = bom.get_best_estimator()
y_pred = clsf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))