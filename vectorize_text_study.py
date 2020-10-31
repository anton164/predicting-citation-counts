import pandas as pd
import streamlit as st
import plotly.express as px
from data_tools import preprocess_text, preprocess_text_col, vectorize_text
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def run_vectorize_text_study(raw_docs):
    df = raw_docs.assign(processed_abstract=preprocess_text_col(raw_docs["Abstract"]))

    n_examples = 2
    st.markdown("### Example of preprocessing {} abstracts".format(n_examples))
    st.table(df[["Abstract", "processed_abstract"]][:n_examples])

    min_df = 0.1
    bow_model, vectorizer = vectorize_text(
        df, "processed_abstract", CountVectorizer(min_df=min_df)
    )
    vocabulary_size = bow_model.shape[1]

    st.markdown(
        """
        Vectorizing text with min_df = {}  
        Vocabulary size: {}
    """.format(
            min_df, vocabulary_size
        )
    )

    st.subheader("Most frequent tokens")
    st.table(bow_model.sum().sort_values(ascending=False)[:10])
