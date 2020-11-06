import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
from .language_tools import detect_language
from sklearn.preprocessing import LabelEncoder


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

def extract_author_prominence_feature(doc, author_map, prominence_threshold = 0):

    # Option 1: Averaged sum of all author citation counts
    author_cols = doc.index.str.startswith("Author_")
    author_ids = doc[author_cols][pd.notnull(doc[author_cols])]
    if (len(author_ids) == 0):
        return 0
    author_prominence = 0
    for author_id in author_ids:
        author_prominence += author_map[author_id]["TotalCitationCount"] - author_map[author_id]["CitationCounts"][doc["PaperId"]]
    # return author_prominence / len(author_ids)

    # Option 2: Citation count of main author
    # author_id = doc["Author_0"]
    # author_prominence = author_map[author_id]["TotalCitationCount"] - author_map[author_id]["CitationCounts"][doc["PaperId"]]
    # return author_prominence

    # Option 3 Binary prominent author feature
    return 1 if (author_prominence > prominence_threshold) else 0

@st.cache
def add_author_prominence_feature(df, author_map):
    author_prominence_column = df.apply(lambda doc: extract_author_prominence_feature(doc, author_map, 50), axis=1)
    df_with_author_prominence = df.assign(AuthorProminence=author_prominence_column)

    return df_with_author_prominence

@st.cache
def add_magbin_feature(df):
    label_encoder = LabelEncoder()
    df["MagBin"] = label_encoder.fit_transform(pd.cut(df.Rank, 4, retbins=True)[0])

    return df

@st.cache
def add_citationbin_feature(df):
    label_encoder = LabelEncoder()
    df["CitationBin"] = label_encoder.fit_transform(pd.cut(df.CitationCount, 4, retbins=True)[0])

    return df