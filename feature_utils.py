import streamlit as st
from utils import detect_language, time_it

detect_language = time_it("Detecting language", detect_language)


@st.cache
def add_language_feature(df):
    language_column = detect_language(df["Abstract"])
    df_with_language = df.assign(Language=language_column)

    return df_with_language