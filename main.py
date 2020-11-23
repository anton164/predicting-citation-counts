import time
import streamlit as st
from interface import (
    components,
    feature_selection,
)
from data_tools import (
    st_dataset_selector,
    load_dataset,
    time_it,
)

# Wrap methods with timer:
load_dataset = time_it(
    lambda ret: "Loading dataset ({} docs)".format((len(ret[0]))),
    load_dataset,
)


def feature_selection_page():
    ###############
    # Application Header and data loading
    ###############

    st.header("Data Exploration")
    docs_limit = st.number_input(
        "Max limit of docs to parse (more than 10000 items will be slow)",
        value=1000,
        step=50,
    )
    selected_dataset = st_dataset_selector()

    raw_docs, author_map = load_dataset(
        selected_dataset, docs_limit, use_predefined_filters=True
    )

    ##############
    # Extracts available data columns and creates checkboxes
    ##############
    st.header("Experiment Setup")
    (
        doc_types,
        features,
        derived_features,
        included_languages,
        years_since_publication_limit,
        selected_field_of_study,
    ) = feature_selection.data_selection(raw_docs)

    st.subheader("Part 3: Compile Dataset")
    filename = st.text_input("Filename (Don't add .csv):", "")
    save = components.get_checkboxes(["Save after compiling"])

    df = None
    if st.button("Create Dataset"):
        if save["Save after compiling"]:
            if filename == "":
                filename = str(int(time.time()))
            df = feature_selection.compile_df(
                raw_docs,
                author_map,
                doc_types,
                features,
                derived_features,
                included_languages,
                years_since_publication_limit,
                selected_field_of_study,
                out_file=filename,
            )
            st.write("Successfully saved dataframe to " + filename)
        else:
            df = feature_selection.compile_df(
                raw_docs,
                author_map,
                doc_types,
                features,
                derived_features,
                included_languages,
                years_since_publication_limit,
                selected_field_of_study,
            )


if __name__ == "__main__":
    feature_selection_page()
