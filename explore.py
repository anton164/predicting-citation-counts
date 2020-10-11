import json
import pandas as pd
import streamlit as st
import os
import time

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


# @st.cache(suppress_st_warning=True)
def load_dataset(dataset_filename, limit):
    start_time = time.perf_counter()
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
            del doc["FirstPage"]
            del doc["LastPage"]

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

    print("One-hot encoding authors")
    author_cols = [col for col in df if col.startswith("Author_")]
    df = pd.get_dummies(df, columns=author_cols, sparse=True, prefix="Author")
    print("Finished one-hot encoding")
    print("Took " + str(time.perf_counter() - start_time) + "s   to load the dataset")
    return df


raw_docs = load_dataset(selected_dataset, docs_limit)
loading_bar.empty()

st.subheader("Data shape")
raw_docs.shape

# Only show titles if the data has a lot  dimensions
if raw_docs.shape[0] * raw_docs.shape[1] > 10000:
    st.subheader("First 10 papers")
    st.write(raw_docs.head(10)["Title"])
else:
    st.subheader("First 10 rows")
    st.write(raw_docs.head(10))
