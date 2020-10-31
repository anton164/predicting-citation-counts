import streamlit as st
import numpy as np


def get_string_columns(df, include=[]):
    t = df.loc[:, include] if len(include) else df
    string_df = t.select_dtypes(exclude=np.number)

    return string_df.columns.tolist()


def separate_datasets(df, *args, y_columns=None):
    dfs = []
    for df_cols in args:
        if y_columns:
            df_cols.extend(y_columns)
        dfs.append(df.loc[:, df_cols])

    return dfs
