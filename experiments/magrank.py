import streamlit as st
from .base import Experiment
from data_tools import (
    vectorize_text,
    preprocess_text_col,
    add_language_feature,
    add_author_prominence_feature,
)
from distribution_study import show_distribution
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def one_hot_encode(df, columns):
    df = pd.get_dummies(
        df,
        columns=columns,
    )
    df = df.fillna(0)
    return df


def print_model_results(Y_pred, Y_test):
    mean_absolute_error = np.abs(Y_pred - Y_test).mean()
    mean_percentage_error = (np.abs(Y_pred - Y_test) / Y_test).mean() * 100
    score = r2_score(Y_test, Y_pred)

    st.markdown(
        """
        #### Model results:
        - Mean absolute error: {:,.0f}
        - Mean percentage error: {:,.2f}%
        - R2 Score: {}    
    """.format(
            mean_absolute_error, mean_percentage_error, r2_score(Y_test, Y_pred)
        )
    )

    return score


class MagRankExperiment(Experiment):
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

        #pandas_df = pandas_df[pandas_df["FieldOfStudy_0"] == "medicine"]

        self.X = pandas_df[
                ["Abstract", "FieldOfStudy_0", "FieldOfStudy_1", "AuthorProminence", "CitationCount"]
            ]
        self.y = pandas_df["Rank"]
        self.model = None

    def preprocess(self):
        st.subheader("Before Preprocessing")
        st.write("X shape: " + str(self.X.shape))
        st.write("Y shape: " + str(self.y.shape))

        # df.select_dtypes(exclude=["int", "float"]).columns
        self.X = one_hot_encode(self.X, ["FieldOfStudy_0", "FieldOfStudy_1"])

        st.dataframe(self.X[:5])

        self.X = self.X.assign(Processed_Abstract=preprocess_text_col(self.X["Abstract"]))
        vectorized_text, vectorizer = vectorize_text(
            self.X, "Processed_Abstract", 
            # CountVectorizer(min_df=0.05, max_df=0.8)
            TfidfVectorizer(min_df=0.01, max_df=0.8)
        )
        st.write("Vocabulary size: " + str(vectorized_text.shape[1]))
        self.X = self.X.drop(columns=["Abstract", "Processed_Abstract"])

        self.X = self.X.join(vectorized_text)

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
        # st.write(self.X_train[:5])
        # st.write(self.y_train[:5])

        self.models = [
            ("Ridge Regression", Ridge(1).fit(self.X_train, self.y_train)),
            #("Multinomial Naive Bayes", Mult.fit(self.X_train, self.y_train)),
            ("Linear SVM", LinearSVC(C=0.01, max_iter=80).fit(self.X_train, self.y_train)),
            ("Random Forest", RandomForestRegressor(max_depth=4, random_state=0).fit(self.X_train, self.y_train)),
            ("Gradient Boosting", GradientBoostingRegressor(max_depth=4, random_state=0).fit(self.X_train, self.y_train))
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