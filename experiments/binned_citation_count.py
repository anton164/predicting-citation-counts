import streamlit as st
from .base import Experiment
from data_tools import (
    vectorize_text,
    preprocess_text_col,
    add_language_feature,
    add_author_prominence_feature,
    filter_by_field_of_study,
)
from distribution_study import show_distribution
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from .sklearn_utils import ClfSwitcher
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache(allow_output_mutation=True)
def hyperparameter_tuning(pipeline, parameters):
    return GridSearchCV(
        pipeline, parameters, cv=2, n_jobs=12, return_train_score=False, verbose=3
    )


st.set_option("deprecation.showPyplotGlobalUse", False)
select_columns = lambda features: FunctionTransformer(
    lambda x: x[features], validate=False
)


def print_model_results(score):
    st.markdown(
        """
        #### Model results:
        - Score: {}%
    """.format(
            score * 100
        )
    )


def group_by_column(df, col):
    grouped_by_col = (
        df.groupby([col])
        .size()
        .reset_index(name="countPapers")
        .sort_values("countPapers", ascending=False)
    ).set_index(col)

    grouped_by_col["Percentage"] = (
        100 * grouped_by_col["countPapers"] / grouped_by_col["countPapers"].sum()
    )
    min_values = []
    max_values = []
    for val in grouped_by_col.index:
        min_values.append(df[df[col] == val]["CitationCount"].min())
    for val in grouped_by_col.index:
        max_values.append(df[df[col] == val]["CitationCount"].max())
    grouped_by_col["Min citation count"] = min_values
    grouped_by_col["Max citation count"] = max_values
    grouped_by_col["Percentage"] = (
        100 * grouped_by_col["countPapers"] / grouped_by_col["countPapers"].sum()
    )
    return grouped_by_col.sort_values("Max citation count")


def show_distribution(df, col, render_limit=10):
    grouped_by_column = group_by_column(df, col)
    n_categories = grouped_by_column.shape[0]
    st.subheader("{} distribution".format(col))
    if render_limit and n_categories > render_limit:
        st.write(
            "Showing top {}, there are {} categories in total".format(
                render_limit, n_categories
            )
        )
        st.table(grouped_by_column[:render_limit])
    else:
        st.table(grouped_by_column)


class BinnedCitationCountExperiment(Experiment):
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

        self.pandas_df = pandas_df
        self.model = None

    def preprocess(self, binary_classification=False):
        if binary_classification:
            self.pandas_df["CitationBin"] = self.pandas_df["CitationBin"].replace(
                {"below-average": "low", "above-average": "high"}
            )
        self.X = self.pandas_df[
            [
                "Abstract",
                "FieldOfStudy_0",
                "FieldOfStudy_1",
                "FieldOfStudy_2",
                "FieldOfStudy_3",
                "AuthorProminence",
                "JournalName",
                "Publisher",
                "PageCount",
                "AuthorRank",
                "PublisherRank",
                "JournalNameRank",
            ]
        ]
        self.y = self.pandas_df["CitationBin"]

        st.subheader("After Preprocessing")
        st.write("X shape: " + str(self.X.shape))
        st.write("Y shape: " + str(self.y.shape))
        st.subheader("JournalNameRank")
        self.pandas_df["JournalNameRank"].hist(bins=100)
        st.pyplot()
        st.subheader("AuthorRank")
        self.pandas_df["AuthorRank"].hist(bins=100)
        st.pyplot()
        st.subheader("PublisherRank")
        self.pandas_df["PublisherRank"].hist(bins=100)
        st.pyplot()

        # self.pandas_df[].hist()
        # )

        show_distribution(self.pandas_df, "CitationBin")

    def run(self):
        docs_limit = st.number_input(
            "Limit dataset size (more than 10000 items will be slow)",
            value=1000,
            step=50,
        )
        binary_classification = st.checkbox("Convert to binary classification")

        self.pandas_df = self.pandas_df[:docs_limit]
        self.preprocess(binary_classification)
        self.split(0.15)
        st.write("X_train shape: " + str(self.X_train.shape))
        st.write("y_train shape: " + str(self.y_train.shape))

        if st.button("Train model & evaluate"):
            self.model = self.train()
            self.evaluate()
        pass

    # @st.cache
    def train(self):
        self.model_pipeline = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "numeric",
                                Pipeline(
                                    [
                                        (
                                            "selector",
                                            select_columns(
                                                [
                                                    "AuthorProminence",
                                                    "PageCount",
                                                    "PublisherRank",
                                                    "JournalNameRank",
                                                    # "AuthorRank",
                                                ]
                                            ),
                                        )
                                    ]
                                ),
                            ),
                            (
                                "categorical",
                                Pipeline(
                                    [
                                        (
                                            "selector",
                                            select_columns(
                                                [
                                                    "FieldOfStudy_0",
                                                    # 'FieldOfStudy_1',
                                                    # "FieldOfStudy_2",
                                                    # "JournalName",
                                                    # "Publisher",
                                                ]
                                            ),
                                        ),
                                        # ('label_encoder', LabelEncoder()),
                                        (
                                            "one_hot_encoder",
                                            OneHotEncoder(
                                                sparse=True, handle_unknown="ignore"
                                            ),
                                        ),
                                    ]
                                ),
                            ),
                            # (
                            #     "text",
                            #     Pipeline(
                            #         [
                            #             (
                            #                 "selector",
                            #                 select_columns("Processed_Abstract"),
                            #             ),
                            #             (
                            #                 "tfidf",
                            #                 TfidfVectorizer(
                            #                     min_df=0.05,
                            #                     max_df=0.75,
                            #                     max_features=500,
                            #                 ),
                            #             ),
                            #         ]
                            #     ),
                            # ),
                        ]
                    ),
                ),
                ("clf", ClfSwitcher()),
            ]
        )

        self.pipeline_parameters = [
            {
                "clf__estimator": [MultinomialNB(alpha=2)],
                "clf__estimator__alpha": (0.5, 1, 1.5, 2, 3, 5),
                # 'features__text__tfidf__max_df': (0.25, 0.5, 0.7),
                # 'features__text__tfidf__min_df': (0.01, 0.05, 0.1),
            },
            {
                # 'features__text__tfidf__min_df': (0.01, 0.05, 0.1),
                # 'features__text__tfidf__max_df': (0.25, 0.5, 0.7),
                "clf__estimator": [XGBClassifier(random_state=0)],
            },
            {
                #     # 'features__text__tfidf__max_df': (0.25, 0.5),
                "clf__estimator": [
                    RandomForestClassifier(random_state=0, n_estimators=50)
                ],
                "clf__estimator__max_depth": (25, 50),
            },
        ]

        model = hyperparameter_tuning(self.model_pipeline, self.pipeline_parameters)
        model.fit(self.X_train, self.y_train)
        return model

    def evaluate(self):
        citation_bins = np.unique(self.y)

        st.subheader("Baseline model (random guesses)")
        print_model_results(1.0 / len(citation_bins))

        st.header("Hyper-parameter tuning")

        st.write("Best parameters:")
        st.write(self.model.best_params_)

        count_samples = 40
        model_prediction_labels = ["Truth"]
        model_predictions = [self.y_test[:count_samples].values]

        models = [(self.model.best_params_["clf__estimator"], self.model)]

        for (model_name, model) in models:
            y_pred = model.predict(self.X_test)
            st.subheader(model_name)
            print_model_results(model.score(self.X_test, self.y_test))
            st.markdown(
                (
                    """
**Classification report:**  
```
{}
```
"""
                ).format(classification_report(self.y_test, y_pred))
            )

            st.write("**Confusion matrix:**")
            confusion_matrix_df = pd.DataFrame(
                confusion_matrix(self.y_test, y_pred), columns=citation_bins
            )
            confusion_matrix_df.index = citation_bins
            st.dataframe(confusion_matrix_df)

            model_prediction_labels.append(model_name)
            model_predictions.append(y_pred[:count_samples])

        means = self.model.cv_results_["mean_test_score"]
        stds = self.model.cv_results_["std_test_score"]
        tuning_results = zip(
            map(
                lambda result: result["clf__estimator"],
                self.model.cv_results_["params"],
            ),
            # map(lambda result: result["features__text__tfidf__max_df"], self.model.cv_results_['params']),
            means,
            stds,
        )

        tuning_df = pd.DataFrame(
            data=tuning_results,
            columns=[
                "Estimator",
                # "Max df",
                "Mean",
                "STD",
            ],
        )

        st.write(tuning_df)

        pass
