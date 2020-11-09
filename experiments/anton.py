import streamlit as st
from .base import Experiment
from data_tools import add_language_feature
from distribution_study import show_distribution


class AntonExperiment(Experiment):
    def __init__(self, pandas_df):
        pandas_df = add_language_feature(pandas_df)
        self.df = pandas_df
        self.X = pandas_df.iloc[:, :-1].values
        self.y = pandas_df.iloc[:, -1].values
        self.model = None

    def run(self):
        self.preprocess()
        self.train()
        self.evaluate()
        pass

    def train(self):
        pass

    def evaluate(self):
        show_distribution(self.df, "Language")
