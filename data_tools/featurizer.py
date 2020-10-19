import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def get_page_count(first_page, last_page):
    return [
        int(l) - int(f) if l and f is not None else None
        for f, l in zip(first_page, last_page)
    ]


le = {}
from sklearn.preprocessing import LabelEncoder


def encode_categorical(df, cols):
    """
    NOTE: Only works for "Small sample (50 rows) for now"
    """
    t = df.copy().dropna(subset=cols)
    global le
    for col in cols:
        le[col] = LabelEncoder()
        t[col] = le[col].fit_transform(t[col])

    return t


def decode_categorical(df, cols):
    """
    NOTE: Only works for "Small sample (50 rows) for now"
    """
    t = df.copy()
    global le
    for col in cols:
        if col in le.keys():
            t[col] = le[col].inverse_transform(t.loc[:, col])

    return t
