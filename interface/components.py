import streamlit as st


def get_checkboxes(label_list, num_cols=3):
    out_dict = {}
    cols = st.beta_columns(num_cols)
    for i, label in enumerate(label_list):
        col = cols[i % num_cols]
        chx_bx = col.checkbox(str(label))
        out_dict[label] = chx_bx

    return out_dict
