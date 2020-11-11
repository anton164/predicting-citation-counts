import pandas as pd
import streamlit as st
from data_tools import st_dataset_selector, load_dataset, filter_by_field_of_study


def find_authors_by_name(query, author_map):
    authors = []

    if len(query) > 3:
        for author in author_map.values():
            if query.lower() in author["Name"].lower():
                authors.append(author)
    return authors


def paper_exploration_page():

    docs_limit = st.number_input(
        "Max limit of docs to parse (more than 10000 items will be slow)",
        value=1000,
        step=50,
    )
    selected_dataset = st_dataset_selector()

    raw_docs, author_map = load_dataset(selected_dataset, docs_limit)

    # raw_docs = filter_by_field_of_study(raw_docs, "computer science")

    author_name = st.text_input("Search for author")
    authors = pd.DataFrame(find_authors_by_name(author_name, author_map))

    st.subheader("{} authors found".format(len(authors)))

    if (len(authors) > 0):
        st.dataframe(
            authors
        )

        if (len(authors) < 5):
            for index, author in authors.iterrows():
                print(author)
                st.markdown("""
                    ### {}  
                    **CitationCounts [PaperId: Count]:**   

                    {}
                """.format(author["Name"], author["CitationCounts"]))

        paper_id_input = st.text_input("Select paper by id:")
        paper_id = int(paper_id_input) if paper_id_input.isdigit() else None
        found_papers = raw_docs[raw_docs["PaperId"] == paper_id]

        if len(found_papers) > 0:
            paper = found_papers.iloc[0]
            print(paper)
            st.subheader(paper["Title"])
            st.dataframe(paper)
            st.markdown(
                """
                **Field of Study:** _{}_  
                **Abstract:**  

                {}
            """.format(
                    paper["FieldOfStudy_0"], paper["Abstract"]
                )
            )


if __name__ == "__main__":
    paper_exploration_page()
