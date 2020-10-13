import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np


def describe(pd_series, title="New Series Analysis", xlabel=None):
    st.subheader(f"{title}")
    st.write(f"\t range: [{pd_series.min()}, {pd_series.max()}]")
    st.write(f"\t mean, std: [{pd_series.mean().round(2)}, {pd_series.std().round(2)}]")
    st.write(f"\t median: {pd_series.median()}")
    pd_hist = plt.figure()
    pd_series.plot(kind='hist')
    plt.xlabel(f"{xlabel if xlabel else title}")
    st.pyplot(pd_hist)

def correlation(df, columns=None, plot = False):
    t = df.loc[:,columns] if columns else df
    if not plot:
        st.write(t.corr())
    else:
        fig = plt.figure()
        mask = np.triu(t.corr())
        sns.heatmap(t.corr(), mask=mask, cbar=False, square=True, annot=True, vmin=-1, vmax=1, center=0, cmap="coolwarm_r", linecolor="black")
        st.pyplot(fig)


def show_relative_scatter(df, x, y):
    fig = plt.figure()
    plt.scatter(df.loc[:,x].values, df.loc[:,y].values)
    plt.title(f"{' vs. '.join([x, y])}")
    plt.xlabel(x)
    plt.ylabel(y)
    st.pyplot(fig)