from numpy.core.numeric import outer
import pandas as pd
import streamlit as st
from data_tools import (
    separate_datasets,
    st_dataset_selector,
    load_dataset,
    time_it,
    one_hot_encode_authors,
)

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



configs = {
    "studies": [
        "run_correlation_study",

    ],
}

##############
# Extracts available data columns and creates checkboxes
##############
feature_list = raw_docs.columns.tolist()
feature_list.sort()


def get_checkboxes(label_list, num_cols = 3):
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

st.subheader("Part 2: Attribute selection")
feature_list = [f for f in raw_docs.columns.tolist() if f != "DocType"]
feature_list.sort()
features = get_checkboxes(feature_list)

st.subheader("Part 3: Compile Dataset")
df = None
if st.button("Create Dataset"):
    selected_types = [t for t in doc_type_list if doc_types[t]]
    selected_features = [f for f in feature_list if features[f]]
    
    col1, col2 = st.beta_columns(2)
    col1.write("Selected Document Types:")
    col2.write(", ".join(selected_types))
    col1.write("Selected Features:")
    col2.write(", ".join(selected_features))
    
    df = raw_docs.copy()
    for t in doc_type_list:
        if not doc_types[t]:
            df.drop(df[df.DocType == t].index, inplace=True)

    df = separate_datasets(df, selected_features)[0]

    st.subheader("Raw docs shape")
    st.write(df.shape)

    st.subheader("First 10 papers")
    st.write(df.head(10))

st.write("Part 4: Study selection")


if df is not None and st.button("Run New Study"):
    something = 10






# st.subheader("Features")
# st.write(", ".join(raw_docs.columns))

#############
# Drop down of available studies to run
#############

##############
# Doctype selector and separate data by selected doctype
##############

#############
# Number of experiment containers to load (Hide in sidebar)
#############

##############
# Render checkboxes for every experiment container
##############








# from correlation_study import run_correlation_study

# run_correlation_study(raw_docs)


# from distribution_study import run_distribution_study

# run_distribution_study(raw_docs)

# from vectorize_text_study import run_vectorize_text_study

# st.markdown(
#     """
#     ## Vectorizing abstracts
#     **Warning:** This is slow for n > 10 000 docs
#     """
# )
# if st.button("Vectorize text"):
#     run_vectorize_text_study(raw_docs)

# st.markdown(
#     """
#     ## One-hot encoding authors
#     **Warning:** This is slow on n > 10 000 docs
#     """
# )
# if st.button("Run one-hot encoding"):
#     one_hot_encoded = one_hot_encode_authors(raw_docs)
#     st.subheader("One-hot-encoded authors shape")
#     one_hot_encoded.shape
