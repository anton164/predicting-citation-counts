import pandas as pd
import plotly.express as px
import spacy
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words as englishwords
from nltk.stem import WordNetLemmatizer

# if this throws and error do: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
STOPWORDS = nlp.Defaults.stop_words


def include_token(token):
    return (
        not token.is_punct
        and not token.like_num
        and token.lemma_ != "-PRON-"
        and token.lemma_ not in STOPWORDS
        and not token.is_space
    )


def preprocess_text(text):
    # Lower-case
    text = text.lower()

    # Lemmatize and filter tokens
    text_doc = nlp(text)
    lemmatized = [token.lemma_ for token in text_doc if include_token(token)]

    return " ".join(lemmatized)


def preprocess_text_col(text_col):
    preproc_pipe = []
    for doc in nlp.pipe(
        text_col,
        batch_size=20,
        disable=["tagger", "parser", "entity", "ner", "textcat"],
    ):
        tokens = [token.lemma_ for token in doc if include_token(token)]
        preproc_pipe.append((" ".join(tokens).lower()))
    return preproc_pipe


def vectorize_text(df, text_col, vectorizer):
    vectorized = vectorizer.fit_transform(df[text_col])
    vectorized_df = pd.DataFrame.sparse.from_spmatrix(
        vectorized, columns=vectorizer.get_feature_names(), index=df.index
    )

    return vectorized_df, vectorizer


def tokenize(
    string, lower=True, punctuation=True, stops=True, lemmatize=True, check_english=True
):
    """Tokenise a string into a vector"""
    stop_words = set(stopwords.words("english"))
    english_words = set(englishwords.words())
    words = word_tokenize(string)
    lzr = WordNetLemmatizer()

    # Remove Lower Case
    if lower:
        words = [w.lower() for w in words]
    # Remove Punctuation
    if punctuation:
        words = [w for w in words if w.isalnum()]
    # Remove Stop Words
    if stops:
        words = [w for w in words if w not in stop_words]
    # Lemmatise
    if lemmatize:
        words = [lzr.lemmatize(w) for w in words]
    # Check if word is Enlgish; Takes long to run, use with caution
    if check_english:
        words = [w for w in words if w in english_words]
    return words
