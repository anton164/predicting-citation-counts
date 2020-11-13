from main import feature_selection_page
from experiment_selection import experiment_selection_page
from distribution_study import distribution_study_page
from paper_exploration import paper_exploration_page
import streamlit as st

PAGES = {
    "Paper Exploration": paper_exploration_page,
    "Feature Selection": feature_selection_page,
    "Experiment Selection": experiment_selection_page,
    "Distribution Study": distribution_study_page,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page()
