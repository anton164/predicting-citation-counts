import streamlit as st
import pycld2 as cld2
from .perf_utils import time_it


def cld2_detect_language(text):
    """
    Detect language of text using cld2
    See https://pypi.org/project/pycld2/ for an example
    """
    try:
        isReliable, textBytesFound, details = cld2.detect(text)
    except:
        st.header("Failed to detect language for:")
        st.write(text)
        raise

    return details[0]


@st.cache
def detect_language(df_column):
    return df_column.apply(lambda x: cld2_detect_language(x)[0])


detect_language = time_it("Detecting language", detect_language)
