import streamlit as st
from .base import Experiment


class TestExperiment(Experiment):
    def run(self):
        st.write("Ran Test experiment for X: " + str(self.X.shape))
        pass