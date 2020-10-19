import streamlit as st
import data_tools as dt


def run_correlation_study(raw_docs):
    st.title("Correlation Study")

    st.header("1. Dependent Variable Analysis")
    dt.describe(raw_docs.Rank, title="Rank", xlabel="Rank (discrete)")
    dt.describe(
        raw_docs.CitationCount,
        title="Citation Count",
        xlabel="Citation Count (discrete)",
    )

    st.subheader("Rank vs. Citation Count (Small correlation)")
    dt.correlation(raw_docs, ["Rank", "CitationCount"])
    dt.show_relative_scatter(raw_docs, "Rank", "CitationCount")

    st.header("2. Independent Variable Analysis")
    st.subheader("DocType")
    df = raw_docs.copy()
    df["PageCount"] = dt.get_page_count(df["FirstPage"].values, df["LastPage"].values)

    journals = df[df["DocType"] == "Journal"]
    dt.describe(journals.Rank, title="Journal Rank", xlabel="Rank (discrete)")
    dt.describe(
        journals.CitationCount,
        title="Journal Citation Count",
        xlabel="Citation Count (discrete)",
    )

    books = df[df["DocType"] == "Book"]
    dt.describe(books.Rank, title="Book Rank", xlabel="Rank (discrete)")
    dt.describe(
        books.CitationCount,
        title="Book Citation Count",
        xlabel="Citation Count (discrete)",
    )

    patents = df[df["DocType"] == "Patent"]
    dt.describe(patents.Rank, title="Patent Rank", xlabel="Rank (discrete)")
    dt.describe(
        patents.CitationCount,
        title="Patent Citation Count",
        xlabel="Citation Count (discrete)",
    )

    conference_papers = df[df["DocType"] == "Conference"]
    dt.describe(
        conference_papers.Rank, title="Conference Rank", xlabel="Rank (discrete)"
    )
    dt.describe(
        conference_papers.CitationCount,
        title="Conference Citation Count",
        xlabel="Citation Count (discrete)",
    )

    st.subheader("Correlation Analysis - With Journal dataset")
    field_of_study = [col for col in raw_docs if col.startswith("FieldOfStudy_")]
    str_cols = dt.get_string_columns(
        journals,
        include=["Publisher", "JournalName", *field_of_study[:3]],
    )
    le_dic = {}
    journals = dt.encode_categorical(journals, str_cols, le_dic)
    st.write(journals.head())

    y_columns = ["Rank", "CitationCount"]
    df_0, df_1, df_2, df_3 = dt.separate_datasets(
        journals,
        [],
        ["JournalName", "Publisher", "FirstPage", "LastPage", *field_of_study[:3]],
        ["PageCount", *field_of_study[:3]],
        ["Title", "Abstract"],
        y_columns=y_columns,
    )

    for df in [df_1, df_2]:
        dt.correlation(df, plot=True)
