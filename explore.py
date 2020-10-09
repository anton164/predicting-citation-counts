import pandas as pd
import streamlit as st 



st.header("Exploring the dataset")

raw_docs = pd.read_json("./sample_data.jsonl", lines=True)

st.write(raw_docs)

st.write(raw_docs.describe())