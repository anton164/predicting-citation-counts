import streamlit as st
import plotly.express as px
from data_tools import (
    add_language_feature,
    load_dataset,
    st_dataset_selector,
    filter_by_field_of_study,
)
import numpy as np


def group_by_column(df, col):
    grouped_by_col = (
        df.groupby([col])
        .size()
        .reset_index(name="countPapers")
        .sort_values("countPapers", ascending=False)
    ).set_index(col)

    grouped_by_col["Percentage"] = (
        100 * grouped_by_col["countPapers"] / grouped_by_col["countPapers"].sum()
    )
    return grouped_by_col


def show_distribution(df, col, render_limit=10):
    grouped_by_column = group_by_column(df, col)
    n_categories = grouped_by_column.shape[0]
    st.subheader("{} distribution".format(col))
    if render_limit and n_categories > render_limit:
        st.write(
            "Showing top {}, there are {} categories in total".format(
                render_limit, n_categories
            )
        )
        st.table(grouped_by_column[:render_limit])
    else:
        st.table(grouped_by_column)


def run_distribution_study(raw_docs):
    df = add_language_feature(raw_docs)
    st.header("Distribution of papers (by categorical features)")

    show_distribution(df, "FieldOfStudy_0")
    df = filter_by_field_of_study(df)
    show_distribution(df, "DocType")

    st.write("Number of papers: " + str(df.shape[0]))

    st.subheader("Rank distribution")
    f = px.histogram(df, x="Rank", nbins=50)
    f.update_yaxes(title="Number of papers")
    st.plotly_chart(f)

    specific_field_of_study = st.selectbox(
        "Inspect more specific field of studies",
        [
            "FieldOfStudy_1",
            "FieldOfStudy_2",
            "FieldOfStudy_3",
            "FieldOfStudy_4",
            "FieldOfStudy_5",
        ],
    )
    show_distribution(df, specific_field_of_study)
    show_distribution(df, "Language")
    show_distribution(df, "Publisher")
    show_distribution(df, "JournalName")


def distribution_study_page():
    docs_limit = st.number_input(
        "Max limit of docs to parse (more than 10000 items will be slow)",
        value=1000,
        step=50,
    )
    selected_dataset = st_dataset_selector()
    raw_docs, author_map = load_dataset(selected_dataset, docs_limit)
    run_distribution_study(raw_docs)
