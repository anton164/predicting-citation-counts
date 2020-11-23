# Streamlit
import streamlit as st
from .base import Experiment
from data_tools import preprocess_text_col

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Data
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score


def print_model_results(Y_pred, Y_test):
    report = classification_report(Y_test, Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)
    st.markdown(f"#### Model results:\n{report}")
    return accuracy


class JournalExperiment(Experiment):
    def __init__(self, pandas_df):
        # Drop non-numeric page features and add PageCount feature
        pandas_df[["FirstPage", "LastPage"]] = pandas_df[
            ["FirstPage", "LastPage"]
        ].apply(pd.to_numeric, errors="coerce")
        pandas_df = pandas_df.dropna(subset=["FirstPage", "LastPage"])
        pandas_df = pandas_df.assign(
            PageCount=pandas_df.apply(
                lambda doc: doc["LastPage"] - doc["FirstPage"], axis=1
            )
        )

        # Consider only medical papers
        pandas_df = pandas_df[pandas_df["FieldOfStudy_0"] == "medicine"]

        # Take AuthorProminence, CitationCount, FieldOfStudy and encode
        categorials = pandas_df[
            ["FieldOfStudy_1", "MagBin", "CitationBin", "Publisher"]
        ]
        encoded_categories = pd.get_dummies(categorials, dummy_na=True)

        # Add encoded cats back
        self.X = pandas_df.drop(
            columns=[
                "CitationCount",
                "FieldOfStudy_0",
                "FieldOfStudy_1",
                "JournalName",
                "FirstPage",
                "LastPage",
                "Publisher",
                "MagBin",
                "Rank",
                "CitationBin",
                "YearsSincePublication",
            ]
        )
        self.X = pd.merge(self.X, encoded_categories, left_index=True, right_index=True)

        # Set target
        self.y = pandas_df["JournalName"]
        self.model = None

    def preprocess(self):
        st.subheader("Preprocessing")
        st.write("X shape: " + str(self.X.shape))
        st.write("Y shape: " + str(self.y.shape))

        self.X = self.X.assign(
            Processed_Abstract=preprocess_text_col(self.X["Abstract"])
        )
        self.X = self.X.fillna("None")
        self.X = self.X.drop(columns="Abstract")
        encoded_words = pd.get_dummies(self.X.Processed_Abstract, dummy_na=True)
        self.X = pd.merge(self.X.drop(columns="Processed_Abstract"), encoded_words, left_index=True, right_index=True)

        st.subheader("After Preprocessing")
        st.write("X shape: " + str(self.X.shape))
        st.write("Y shape: " + str(self.y.shape))

    def run(self):
        self.preprocess()
        self.split(0.15)
        self.train()
        self.evaluate()
        pass

    def train(self):
        st.write("X_train shape: " + str(self.X_train.shape))
        st.write("y_train shape: " + str(self.y_train.shape))
        st.write(self.X_train[:5])
        st.write(self.y_train[:5])

        self.models = [
            (
                "Logistic Regression",
                LogisticRegression().fit(self.X_train, self.y_train),
            ),
            (
                "Linear SVM",
                LinearSVC(C=0.1, max_iter=50).fit(self.X_train, self.y_train),
            ),
            ("Random Forest", RandomForestClassifier().fit(self.X_train, self.y_train)),
            ("XGBoost", XGBClassifier().fit(self.X_train, self.y_train)),
        ]

    def evaluate(self):

        count_samples = 10
        model_prediction_labels = ["Truth"]
        model_predictions = [self.y_test[:count_samples].values]

        for (model_name, model) in self.models:
            y_pred = model.predict(self.X_test)
            st.subheader(model_name)
            print_model_results(y_pred, self.y_test)

            model_prediction_labels.append(model_name)
            model_predictions.append(y_pred[:count_samples])

        st.dataframe(
            pd.DataFrame(
                data=np.array(model_predictions).reshape(
                    count_samples, len(model_predictions)
                ),
                columns=model_prediction_labels,
            )
        )

        pass
