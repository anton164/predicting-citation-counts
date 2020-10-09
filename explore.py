import pandas as pd
import streamlit as st 



st.header("Exploring the dataset")

@st.cache
def load_data():
    return pd.read_json("./sample_data.jsonl", lines=True)

raw_docs = load_data()

st.write(raw_docs.head(10))

st.write(raw_docs.describe())