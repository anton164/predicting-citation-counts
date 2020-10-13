import json
import pandas as pd
import streamlit as st
<<<<<<< HEAD
import os
import time
=======
import matplotlib.pyplot as plt
>>>>>>> My data analysis

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
            # del doc["FirstPage"]
            # del doc["LastPage"]

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

st.subheader("First 10 rows")
st.write(raw_docs.head(10))

st.title("Jan's Data Exploration")
import data_tools as dt

st.header("1. Dependent Variable Analysis")
dt.describe(raw_docs.Rank, title="Rank", xlabel="Rank (discrete)")
dt.describe(
    raw_docs.CitationCount, title="Citation Count", xlabel="Citation Count (discrete)"
)

st.subheader("Rank vs. Citation Count (Small correlation)")
dt.correlation(raw_docs, ["Rank", "CitationCount"])
dt.show_relative_scatter(raw_docs, "Rank", "CitationCount")


st.header("2. Independent Variable Analysis")
st.subheader("DocType")
df = raw_docs.copy()
df["PageCount"] = dt.get_page_count(df["FirstPage"].values, df["LastPage"].values)

journals = df[df["DocType"] == "Journal"]
dt.describe(journals.Rank, title="Journal Rank", xlabel="Rank (discrete)")
dt.describe(
    journals.CitationCount,
    title="Journal Citation Count",
    xlabel="Citation Count (discrete)",
)

books = df[df["DocType"] == "Book"]
dt.describe(books.Rank, title="Book Rank", xlabel="Rank (discrete)")
dt.describe(
    books.CitationCount, title="Book Citation Count", xlabel="Citation Count (discrete)"
)

patents = df[df["DocType"] == "Patent"]
dt.describe(patents.Rank, title="Patent Rank", xlabel="Rank (discrete)")
dt.describe(
    patents.CitationCount,
    title="Patent Citation Count",
    xlabel="Citation Count (discrete)",
)

conference_papers = df[df["DocType"] == "Conference"]
dt.describe(conference_papers.Rank, title="Conference Rank", xlabel="Rank (discrete)")
dt.describe(
    conference_papers.CitationCount,
    title="Conference Citation Count",
    xlabel="Citation Count (discrete)",
)

st.subheader("Correlation Analysis - With Journal dataset")
field_of_study = [col for col in raw_docs if col.startswith("FieldOfStudy_")]
str_cols = dt.get_string_columns(
    journals,
    include=["Publisher", "JournalName", *field_of_study[:3]],
)
journals = dt.encode_categorical(journals, str_cols)
st.write(journals.head())

y_columns = ["Rank", "CitationCount"]
df_0, df_1, df_2, df_3 = dt.separate_datasets(
    journals,
    [],
    ["JournalName", "Publisher", "FirstPage", "LastPage", *field_of_study[:3]],
    ["PageCount", *field_of_study[:3]],
    ["Title", "Abstract"],
    y_columns=y_columns,
)

for df in [df_1, df_2]:
    dt.correlation(df, plot=True)
