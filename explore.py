import json
import pandas as pd
import streamlit as st
import os
import plotly.express as px
from utils import time_it

st.header("Data Exploration")

# Enables streamlit to read data from mounted disk in GCP
dataset_location = "./"
if "STREAMLIT_DATA_LOCATION" in os.environ:
    dataset_location = os.environ["STREAMLIT_DATA_LOCATION"]

datasets = {
    dataset_location + "sample_data.jsonl": "Small sample (50 rows)",
    dataset_location + "250k.docs.jsonl": "Large sample (250k rows)",
    dataset_location + "mag5.docs.jsonl": "Full dataset (5m rows, slooow)",
}

selected_dataset = st.selectbox(
    "Selected Dataset",
    list(datasets.keys()),
    format_func=lambda x: datasets[x],
)

docs_limit = st.number_input(
    "Max limit of docs to parse (more than 10000 items will be slow)",
    value=1000,
    step=50,
)


loading_bar = st.progress(0)


@st.cache(suppress_st_warning=True)
def load_dataset(dataset_filename, limit):
    json_data = []
    i = 0
    print("Loading dataset")
    with open(dataset_filename) as file:
        for json_line in file:
            doc = json.loads(json_line)

            # Extract author id (we don't care about AuthorName and SequenceNumber for now)
            for k, author in enumerate(doc["Authors"]):
                doc["Author_" + str(k + 1)] = author["AuthorId"]
            del doc["Authors"]

            # Map fields of study
            for field_of_study in doc["FieldsOfStudy"]:
                doc["FieldOfStudy_" + str(field_of_study["Level"])] = field_of_study[
                    "Name"
                ]
            del doc["FieldsOfStudy"]

            # Extract JournalName from Journal (also contains JournalId, Website)
            if doc["Journal"]:
                doc["JournalName"] = doc["Journal"]["JournalName"]
            del doc["Journal"]

            # For now we don't care about these columns
            del doc["Urls"]
            del doc["PdfUrl"]
            del doc["Doi"]
            del doc["BookTitle"]
            del doc["Volume"]
            del doc["Issue"]

            json_data.append(doc)
            i += 1

            if i % 50 == 0:
                loading_bar.progress(i / limit)

            if i >= limit:
                loading_bar.progress(100)
                loading_bar.empty()
                break

    loading_bar.empty()
    print("Finished loading the data")
    dataframe_loader = st.spinner("Loading dataframe")
    df = pd.DataFrame(json_data)
    print("Created DataFrame")
    return df


def one_hot_encode_authors(df):
    author_cols = [col for col in df if col.startswith("Author_")]
    df = pd.get_dummies(df, columns=author_cols, sparse=True, prefix="Author")
    return df


# Wrap methods with timer:
load_dataset = time_it(
    lambda df: "Loading dataset ({} docs)".format((len(df))),
    load_dataset,
)
one_hot_encode_authors = time_it("One-hot encoding authors", one_hot_encode_authors)


raw_docs = load_dataset(selected_dataset, docs_limit)
loading_bar.empty()

st.subheader("Raw docs shape")
raw_docs.shape

st.subheader("First 10 papers")
st.write(raw_docs.head(10))

from correlation_study import run_correlation_study

run_correlation_study(raw_docs)

st.markdown(
    """
    ## Distribution of papers
    """
)

st.markdown(
    """
    ### Rank
    """
)
f = px.histogram(raw_docs, x="Rank", title="Rank distribution", nbins=50)
f.update_yaxes(title="Count")
st.plotly_chart(f)

st.markdown(
    """
    ### Field of study
    """
)

# f = px.bar(raw_docs, x="FieldOfStudy_0")
# st.plotly_chart(f)


st.markdown(
    """
    ## One-hot encoding authors
    **Warning:** This is slow on large datasets
    """
)
if st.button("Run one-hot encoding"):
    one_hot_encoded = one_hot_encode_authors(raw_docs)
    st.subheader("One-hot-encoded authors shape")
    one_hot_encoded.shape
