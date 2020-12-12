import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


## Constants
DATA = "../saved/data_dec_11_morning.csv"

FEATURE = [    
    # "PaperId",
    "JournalNameRankNormalized",
    "PublisherRankNormalized",
    "AuthorRank",
    "AuthorProminence",
    "PageCount"
]
TARGET = "BinnedCitationsPerYear"


## Data Import
data = pd.read_csv(DATA)

def vectorize_text(df, text_col, vectorizer):
    vectorized = vectorizer.fit_transform(df[text_col])
    vectorized_df = pd.DataFrame.sparse.from_spmatrix(
        vectorized, columns=vectorizer.get_feature_names(), index=df.index
    )

    return vectorized_df, vectorizer


def bin_citation(citation_counts, theta):
    theta_num = int(theta*len(citation_counts))
    sort_counts = citation_counts.sort_values(ascending=False)
    bins = sort_counts.copy()
    bins[:] = -1
    bins[0:theta_num] = 1
    bins[bins.size-theta_num-1:] = 0
    return bins, theta_num

bins, theta_num = bin_citation(data.CitationCountPerYear, 0.10)
data["bin"] = bins
df_uniform  = data[data.bin >= 0]
df_bow, vectorizer = vectorize_text(
        df_uniform, "Processed_Abstract", CountVectorizer(min_df=0.01, max_df=0.5)
    )

low_bow = df_bow[df_uniform.BinnedCitationsPerYear==0]
high_bow = df_bow[df_uniform.BinnedCitationsPerYear==1]
l = np.mean(low_bow.values, axis=0)
h = np.mean(high_bow.values, axis=0)

diff = h - l
abs_diff = np.abs(diff) / (0.5 * (h + l))
vocab_cols = np.argsort(abs_diff)[-50:]


vocab = low_bow.columns.tolist()
top10 = np.argsort(diff[vocab_cols])[-10:]
bottom10 = np.argsort(diff[vocab_cols])[:10]


print(low_bow.iloc[:, vocab_cols].columns.tolist())
print(low_bow.iloc[:, vocab_cols[top10]].columns.tolist())
print(low_bow.iloc[:, vocab_cols[bottom10]].columns.tolist())

features = vectorizer.transform(data.Processed_Abstract)
features = pd.DataFrame.sparse.from_spmatrix(
        features, columns=vectorizer.get_feature_names(), index=data.index
    )

features = features.iloc[:, vocab_cols]