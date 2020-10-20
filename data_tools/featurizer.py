import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from .language_tools import detect_language


def get_page_count(first_page, last_page):
    return [
        int(l) - int(f) if l and f is not None else None
        for f, l in zip(first_page, last_page)
    ]


from sklearn.preprocessing import LabelEncoder


def encode_categorical(df, cols, le_dic):
    """
    NOTE: Only works for "Small sample (50 rows)" for now
    """
    t = df.copy().dropna(subset=cols)
    for col in cols:
        le_dic[col] = LabelEncoder()
        t[col] = le_dic[col].fit_transform(t[col])

    return t


def decode_categorical(df, cols, le_dic):
    """
    NOTE: Only works for "Small sample (50 rows)" for now
    """
    t = df.copy()
    for col in cols:
        if col in le_dic.keys():
            t[col] = le_dic[col].inverse_transform(t.loc[:, col])

    return t


@st.cache
def add_language_feature(df):
    language_column = detect_language(df["Abstract"])
    df_with_language = df.assign(Language=language_column)

    return df_with_language


@st.cache
def one_hot_encode_authors(df):
    author_cols = [col for col in df if col.startswith("Author_")]
    df = pd.get_dummies(df, columns=author_cols, sparse=True, prefix="Author")
    return df