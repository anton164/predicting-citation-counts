import pandas as pd
import streamlit as st 



st.header("Exploring the dataset")

raw_docs = pd.read_json("./250k.docs.jsonl", lines=True)

st.write(raw_docs.head(5))