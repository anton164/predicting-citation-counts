import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import (
    train_test_split,
)
from .base import Experiment


class TestExperiment(Experiment):
    def run(self):
        st.write("Ran Test experiment for X: " + str(self.X.shape))
        pass