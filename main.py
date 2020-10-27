from math import exp
from numpy.core.numeric import outer
import time
import pandas as pd
import streamlit as st
from data_tools import (
    separate_datasets,
    st_dataset_selector,
    load_dataset,
    time_it,
    one_hot_encode_authors,
)
from interface import (
    components,
    feature_selection,
)
import experiments

###############
# Application Header and data loading
###############

st.header("Data Exploration")
# Wrap methods with timer:
load_dataset = time_it(
    lambda df: "Loading dataset ({} docs)".format((len(df))),
    load_dataset,
)
one_hot_encode_authors = time_it("One-hot encoding authors", one_hot_encode_authors)

docs_limit = st.number_input(
    "Max limit of docs to parse (more than 10000 items will be slow)",
    value=1000,
    step=50,
)
selected_dataset = st_dataset_selector()

raw_docs = load_dataset(selected_dataset, docs_limit)


##############
# Extracts available data columns and creates checkboxes
##############
st.header("Experiment Setup")
doc_types, features, dependent_features = feature_selection.data_selection(raw_docs)


st.subheader("Part 3: Compile Dataset")
filename = st.text_input("Filename (Don't add .csv):", "")
save = components.get_checkboxes(["Save after compiling"])


df = None
if st.button("Create Dataset"):
    if save["Save after compiling"]:
        if filename == "":
            filename = str(int(time.time()))
            df = feature_selection.compile_df(
                raw_docs, doc_types, features, dependent_features, out_file=filename
            )
    else:
        df = feature_selection.compile_df(
            raw_docs, doc_types, features, dependent_features
        )
