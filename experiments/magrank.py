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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from .sklearn_utils import ClfSwitcher
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


@st.cache(allow_output_mutation = True)
def hyperparameter_tuning(pipeline, parameters):
    return GridSearchCV(pipeline, parameters, cv=5, n_jobs=12, return_train_score=False, verbose=3)

select_columns = lambda features: FunctionTransformer(lambda x: x[features], validate=False)

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

        self.X = self.X.assign(Processed_Abstract=preprocess_text_col(self.X["Abstract"]))
        self.X = self.X.fillna('None')

        st.subheader("After Preprocessing")
        st.write("X shape: " + str(self.X.shape))
        st.write("Y shape: " + str(self.y.shape))

    def run(self):
        self.preprocess()
        self.split(0.15)
        st.write("X_train shape: " + str(self.X_train.shape))
        st.write("y_train shape: " + str(self.y_train.shape))
        self.model = self.train()
        self.evaluate()
        pass
    
    @st.cache
    def train(self):
        self.model_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('numeric', Pipeline([
                    ('selector', select_columns(['CitationCount', 'AuthorProminence']))
                ])),
                ('text', Pipeline([
                    ('selector', select_columns('Processed_Abstract')),
                    ('tfidf', TfidfVectorizer(min_df=0.01, max_df=0.8))
                ])),
                ('categorical', Pipeline([
                    ('selector', select_columns(['FieldOfStudy_0', 'FieldOfStudy_1'])),
                    # ('label_encoder', LabelEncoder()),
                    ('one_hot_encoder', OneHotEncoder(sparse=True, handle_unknown='ignore'))
                ]))
            ])),
            ('clf', ClfSwitcher())
        ])

        self.pipeline_parameters = [{
            'features__text__tfidf__max_df': (0.25, 0.5, 0.75),
            'clf__estimator': [GradientBoostingRegressor(random_state=0)],
            'clf__estimator__max_depth': (1, 10, 25)
        }, {
            'features__text__tfidf__max_df': (0.25, 0.5, 0.75),
            'clf__estimator': [Ridge()],
            'clf__estimator__alpha': (0, 0.1, 1, 10)
        }, {
            'features__text__tfidf__max_df': (0.25, 0.5, 0.75),
            'clf__estimator': [LinearSVC(max_iter = 100)],
            'clf__estimator__C': (0, 0.1, 1, 10)
        }, {
            'features__text__tfidf__max_df': (0.25, 0.5, 0.75),
            'clf__estimator': [RandomForestRegressor(random_state = 0)],
            'clf__estimator__max_depth': (1, 10, 25)
        }, {
            'features__text__tfidf__max_df': (0.25, 0.5, 0.75),
            'clf__estimator': [XGBRegressor()],
        }]

        model = hyperparameter_tuning(self.model_pipeline, self.pipeline_parameters)
        model.fit(self.X_train, self.y_train)
        return model


    def evaluate(self):
        st.header("Hyper-parameter tuning")
        st.write("Best parameters:")
        st.write(self.model.best_params_)

        count_samples = 10
        model_prediction_labels = ["Truth"]
        model_predictions = [self.y_test[:count_samples].values]

        models = [(self.model.best_params_["clf__estimator"], self.model)]

        for (model_name, model) in models:
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

        means = self.model.cv_results_['mean_test_score']
        stds = self.model.cv_results_['std_test_score']
        tuning_results = zip(
            map(lambda result: result["clf__estimator"], self.model.cv_results_['params']), 
            means, 
            stds
        )

        tuning_df = pd.DataFrame(
            data=tuning_results,
            columns=["Estimator", "Mean", "STD"]
        )

        st.write(tuning_df)

        pass