import numpy as np
import pandas as pd
import json

# models
from bag_of_models import BagOfModels
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# data prep
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_selection import RFECV
import util
import df_utils

# model evaluation
from sklearn.metrics import confusion_matrix, classification_report

N_SPLITS = 4
RAND_STATE = 42
metadata_features = df_utils.load_df("./saved/metadata_features.csv")
target = df_utils.load_df("./saved/binned_citations_threshold_2.csv")

bow_features = [
    df_utils.load_df("./saved/final_datasets/bow_50words_1.csv"),
    df_utils.load_df("./saved/final_datasets/bow_50words_2.csv"),
    df_utils.load_df("./saved/final_datasets/bow_50words_5.csv"),
    df_utils.load_df("./saved/final_datasets/bow_50words_10.csv"),
    df_utils.load_df("./saved/final_datasets/bow_50words_20.csv"),
    df_utils.load_df("./saved/final_datasets/bow_50words_40.csv"),
    df_utils.load_df("./saved/final_datasets/bow_50words_50.csv"),
]

bow_features2 = [
    df_utils.load_df("./saved/final_datasets/bow_100ngrams_1.csv"),
    df_utils.load_df("./saved/final_datasets/bow_100ngrams_2.csv"),
    df_utils.load_df("./saved/final_datasets/bow_100ngrams_5.csv"),
    df_utils.load_df("./saved/final_datasets/bow_100ngrams_10.csv"),
    df_utils.load_df("./saved/final_datasets/bow_100ngrams_20.csv"),
    df_utils.load_df("./saved/final_datasets/bow_100ngrams_40.csv"),
    df_utils.load_df("./saved/final_datasets/bow_100ngrams_50.csv"),
]

# Exclude un-normalized rank features
metadata_features = metadata_features.drop(["JournalNameRank", "PublisherRank"], axis=1)

hyperparams = {
    # 'SVC': {
    #     'kernel': 'linear',
    #     'random_state': 0,
    #     'C': 10.0
    # },
    "LinearSVC": False,
    "NeuralNetwork": {"early_stopping": True, "random_state": 0, "alpha": 1.0},
    "NaiveBayes": False,
    "RandomForest": {"max_depth": 10, "min_samples_split": 0.01},
    "XGBoost": {"max_depth": 5},
}

bom = BagOfModels()
bom.hyperparams = hyperparams


def run_trial(bom, features, targets, cv=N_SPLITS):
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


def average_scores(model_scores):
    models = model_scores[0].keys()
    average_model_scores = {}
    for model in models:
        sum_score = 0
        for scores in model_scores:
            sum_score += scores[model]

        average_model_scores[model] = sum_score / len(model_scores)

    return average_model_scores

def get_features_to_drop(features):
    publication_month_features = list(
        features.columns[features.columns.str.startswith("PublicationMonth_") == True]
    )
    other_features = list(
        features.columns[features.columns.str.startswith("PublicationMonth_") != True]
    )
    if (len(publication_month_features) > 0):
        return [publication_month_features] + other_features
    else:
        return other_features

def drop_one_feature_selection(features):
    features_to_drop = get_features_to_drop(features)
    
    results = {}
    print("Drop-one feature selection test")
    for i, excluded_feature in enumerate(["All features"] + features_to_drop):
        if excluded_feature == "All features":
            print("Running with all features")
            feature_selection = features
        else:
            print(
                "({}/{}) Excluding {}".format(
                    i, len(features_to_drop), excluded_feature
                )
            )
            feature_selection = features.drop(excluded_feature, axis=1)

        t_scores, v_scores, test_scores = run_trial(
            bom, feature_selection.iloc[:].values, target.iloc[:].values.ravel()
        )

        print("---")
        print("Average score across {} splits:".format(N_SPLITS))
        t_avg = average_scores(t_scores)
        v_avg = average_scores(v_scores)
        test_avg = average_scores(test_scores)
        print("train", t_avg)
        print("validation", v_avg)
        print("test", test_avg)
        print("---")

        results[str(excluded_feature)] = [
            {"train": t_avg, "validation": v_avg, "test": test_avg}
        ]

    return results


print("All features:")
features = metadata_features
print(list(features.columns))

if False:
    results = drop_one_feature_selection(features)
    with open("./drop_one_feature_selection_results-2.json", "w") as file:
        json.dump(results, file, sort_keys=True, indent=4)

else:
    X = features.iloc[:].values
    y = target.iloc[:].values.ravel()

    classifiers = {
        "RandomForest": RandomForestClassifier(**hyperparams["RandomForest"]),
        "XGBClassifier": XGBClassifier(**hyperparams["XGBoost"]) 
    }

    all_features = list(features.columns)
    results = {
        "all_features": all_features
    }

    for clf_name, classifier in classifiers.items():
        print("Running RFECV with {}".format(clf_name))
    
        feature_selector = RFECV(
            estimator=classifier,
            step=1,
            cv=ShuffleSplit(n_splits=3, test_size=0.10, random_state=0),
        )

        feature_selector.fit(X, y)
        ranked_features = list(zip(feature_selector.ranking_.tolist(), all_features))
        sorted_ranked_features = sorted(ranked_features, key=lambda x: x[0])
        results[clf_name] = {
            "ranked_features": sorted_ranked_features,
            "relevant_features": list(features.columns[feature_selector.support_]),
            "grid_scores": feature_selector.grid_scores_.tolist()
        }

    with open("./rfecv_metadata_results.json", "w") as file:
        json.dump(results, file, sort_keys=True, indent=4)