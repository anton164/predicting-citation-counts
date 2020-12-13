import json

# models
from bag_of_models import BagOfModels

# data prep
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_selection import RFECV
import util
import df_utils

# model evaluation
from sklearn.metrics import confusion_matrix, classification_report

thresholds = [1, 2, 5, 10, 20, 40, 50]
FEATURE_DATA = dict(zip(thresholds, [f"saved/bow_50words_{t}_min_df_5.csv" for t in thresholds]))

TARGET_DATA = "saved/binned_citations_threshold_2.csv"

# features_cols = [
#     "extensive",
#     "overview",
#     "we",
#     "rich",
#     "survey",
#     "availability",
#     "according",
#     "empirical",
#     "firstly",
#     "so",
#     "http",
# ]


def run_trial(bom, features, targets, cv=5):
    X, X_test, y, y_test = train_test_split(features, targets, test_size=0.1)
    X, y = util.get_uniform_version(X, y)
    runs = ShuffleSplit(n_splits=cv, test_size=0.10, random_state=0)
    t_scores = []
    v_scores = []
    test_scores = []
    for train_idx, val_idx in runs.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        bom.fit(X_train, y_train)
        t_scores.append(bom.training_scores_)

        X_val, y_val = X[val_idx], y[val_idx]
        v_scores.append(bom.score(X_val, y_val))

        test_scores.append(bom.score(X_test, y_test))

    return t_scores, v_scores, test_scores


hyperparams = {
    "SVC": {"kernel": "linear", "random_state": 0, "C": 10.0},
    "NeuralNetwork": {"early_stopping": True, "random_state": 0, "alpha": 1.0},
    # 'NaiveBayes': False,
    "RandomForest": {"max_depth": 10, "min_samples_split": 0.01},
    "XGBoost": {"max_depth": 3},
}

bom = BagOfModels()
bom.hyperparams = hyperparams

results = {}
for thrsh, feature_path in FEATURE_DATA.items():
    # features = df_utils.load_df(feature_path)[features_cols].values
    features = df_utils.load_df(feature_path).values
    target = df_utils.load_df(TARGET_DATA).values.ravel()
    t_scores, v_scores, test_scores = run_trial(bom, features, target, cv=1)
    print(v_scores)
    print(test_scores)
    results[f"Threshold_{thrsh}"] = {
        "train": t_scores,
        "val": v_scores,
        "test": test_scores,
    }

with open("./bin_threshold_results.json", "w") as file:
    json.dump(results, file, sort_keys=True, indent=4)
