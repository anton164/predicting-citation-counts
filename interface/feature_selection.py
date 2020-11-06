from data_tools.featurizer import add_citationbin_feature
import streamlit as st
from .components import (
    get_checkboxes,
)
from data_tools import (
    separate_datasets,
    get_saved_data_location,
    add_language_feature,
    add_author_prominence_feature,
    add_magbin_feature
)
import os

SAVED_FILE_DIR = get_saved_data_location()


def data_selection(data):
    st.subheader("Part 1: Type selection")
    doc_type_list = data.DocType.unique().tolist()
    doc_types = get_checkboxes(doc_type_list)

    st.subheader("Part 2a: Feature selection")
    feature_list = [
        f
        for f in data.columns.tolist()
        if f not in ("DocType")
    ]
    feature_list.sort()
    features = get_checkboxes(feature_list)

    st.subheader("Part 2b: Derived features")
    derived_features_labels = ["AuthorProminence", "MagBin", "CitationBin"]
    derived_features = get_checkboxes(derived_features_labels)

    st.subheader("Part 2c: Languages to include")
    langauge_labels = ["ENGLISH", "GERMAN", "FRENCH"]
    included_languages = get_checkboxes(langauge_labels)

    return doc_types, features, derived_features, included_languages


def compile_df(
    data,
    author_map,
    category_dict,
    features_dict,
    derived_features,
    included_languages,
    out_file=None,
):
    selected_types = [str(k) for k, v in category_dict.items() if v]
    selected_features = [k for k, v in features_dict.items() if v]
    derived_features = [k for k, v in derived_features.items() if v]
    included_languages = [k for k, v in included_languages.items() if v]

    print(selected_types)

    col1, col2 = st.beta_columns(2)
    col1.write("Selected Document Types:")
    col2.write(", ".join(selected_types))
    col1.write("Selected Features (X):")
    col2.write(", ".join(selected_features))
    col1.write("Derived Features (X):")
    col2.write(", ".join(derived_features))

    data_with_language = add_language_feature(data)
    data = data[data_with_language["Language"].isin(included_languages)]

    df = data.copy()

    # Add derived features
    if "AuthorProminence" in derived_features:
        df = add_author_prominence_feature(df, author_map)
    if "MagBin" in derived_features:
        df = add_magbin_feature(df)
    if "CitationBin" in derived_features:
        df = add_citationbin_feature(df)

    for k, v in category_dict.items():
        if not v:
            if k:
                df.drop(df[df.DocType == k].index, inplace=True)
            else:
                df.drop(df[df.DocType.isna()].index, inplace=True)

    df = separate_datasets(
        df, selected_features + derived_features, y_columns=None
    )[0]

    st.subheader("Compiled dataframe shape")
    st.write(df.shape)

    st.subheader("First 5 entries")
    st.write(df.head(5))

    if out_file:
        if not os.path.exists(SAVED_FILE_DIR):
            os.mkdir(SAVED_FILE_DIR)
        df.to_csv(
            os.path.join(SAVED_FILE_DIR, f"{out_file}.csv"), index_label="PaperId"
        )
    return df