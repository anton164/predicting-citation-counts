from math import exp
from numpy.core.numeric import outer
import uuid
import pandas as pd
import streamlit as st
from data_tools import (
    separate_datasets,
    st_dataset_selector,
    load_dataset,
    time_it,
    one_hot_encode_authors,
)
import experiments

EXPERIMENTS = {"Example Experiment": "Experiment"}


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
feature_list = raw_docs.columns.tolist()
feature_list.sort()


def get_checkboxes(label_list, num_cols=3):
    out_dict = {}
    cols = st.beta_columns(num_cols)
    for i, label in enumerate(label_list):
        col = cols[i % num_cols]
        chx_bx = col.checkbox(str(label))
        out_dict[label] = chx_bx

    return out_dict


st.header("Experiment Setup")
st.subheader("Part 1: Type selection")
doc_type_list = raw_docs.DocType.unique().tolist()
doc_types = get_checkboxes(doc_type_list)

st.subheader("Part 2a: Independent variable selection")
feature_list = [
    f
    for f in raw_docs.columns.tolist()
    if f not in ("DocType", "Rank", "CitationCount")
]
feature_list.sort()
features = get_checkboxes(feature_list)

st.subheader("Part 2b: Dependent variable selection")
dependent_variable_list = ["Rank", "CitationCount"]
dependent_features = get_checkboxes(dependent_variable_list)

st.subheader("Part 3: Compile Dataset")
df = None
if st.button("Create Dataset"):
    selected_types = [t for t in doc_type_list if doc_types[t]]
    selected_features = [f for f in feature_list if features[f]]
    selected_dependent_features = [
        f for f in dependent_variable_list if dependent_features[f]
    ]

    col1, col2 = st.beta_columns(2)
    col1.write("Selected Document Types:")
    col2.write(", ".join(selected_types))
    col1.write("Selected Features (X):")
    col2.write(", ".join(selected_features))
    col1.write("Selected Target Values (y):")
    col2.write(", ".join(selected_dependent_features))

    df = raw_docs.copy()
    for t in doc_type_list:
        if not doc_types[t]:
            df.drop(df[df.DocType == t].index, inplace=True)

    df = separate_datasets(
        df, selected_features, y_columns=selected_dependent_features
    )[0]

    st.subheader("Raw docs shape")
    st.write(df.shape)

    st.subheader("First 10 papers")
    st.write(df.head(10))


if df is not None:
    st.subheader("Part 4: Study selection")
    experiment_name = st.selectbox(
        "Select Experiment you want to run:", EXPERIMENTS.keys()
    )
    Experiment = experiments.__dict__[EXPERIMENTS[experiment_name]]
    exp_instance = Experiment(df)
    if st.button("Run New Study"):
        exp_instance.run()
