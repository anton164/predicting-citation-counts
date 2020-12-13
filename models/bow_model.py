import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from df_utils import load_df, join_dfs


def vectorize_text(df, text_col, vectorizer):
    vectorized = vectorizer.fit_transform(df[text_col])
    vectorized_df = pd.DataFrame.sparse.from_spmatrix(
        vectorized, columns=vectorizer.get_feature_names(), index=df.index
    )

    return vectorized_df, vectorizer


def bin_citation(citation_counts, theta):
    theta_num = int(theta * len(citation_counts))
    sort_counts = citation_counts.sort_values(ascending=False)
    bins = sort_counts.copy()
    bins[:] = -1
    bins[0:theta_num] = 1
    bins[bins.size - theta_num - 1 :] = 0
    return bins, theta_num


def create_uniform_df(df_full, theta):
    bins, theta_num = bin_citation(df_full.CitationCountPerYear, theta)
    df_full["bin"] = bins

    return df_full[df_full.bin >= 0]


def create_bow_model(df_full, theta, max_vocab_size=50):
    df_uniform = create_uniform_df(df_full, theta)
    df_bow, vectorizer = vectorize_text(
        df_uniform, "Processed_Abstract", CountVectorizer(min_df=0.05, max_df=0.5)
    )

    low_bow = df_bow[df_uniform.BinnedCitations == 0]
    high_bow = df_bow[df_uniform.BinnedCitations == 1]
    l = np.mean(low_bow.values, axis=0)
    h = np.mean(high_bow.values, axis=0)

    diff = h - l
    abs_diff = np.abs(diff) / (0.5 * (h + l))
    vocab_cols = np.argsort(abs_diff)[-max_vocab_size:]

    top10_indices = np.argsort(diff[vocab_cols])[-10:]
    bottom10_indices = np.argsort(diff[vocab_cols])[:10]

    vocab = low_bow.iloc[:, vocab_cols].columns.tolist()

    top_10 = low_bow.iloc[:, vocab_cols[top10_indices]].columns.tolist()
    bottom_10 = low_bow.iloc[:, vocab_cols[bottom10_indices]].columns.tolist()

    features = vectorizer.transform(df_full.Processed_Abstract)
    features = pd.DataFrame.sparse.from_spmatrix(
        features, columns=vectorizer.get_feature_names(), index=df_full.index
    )

    return features.iloc[:, vocab_cols], vocab, top_10, bottom_10


if __name__ == "__main__":
    ## Data Import
    df_full = join_dfs(
        load_df("./saved/final_dataset_all_variables.csv"),
        load_df("./saved/binned_citations_threshold_2.csv"),
    )

    for theta in [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.50]:
        print("Creating bow model for theta {}".format(theta))

        df_bow, vocab, top10, bottom10 = create_bow_model(df_full, theta)

        print("Vocab")
        print(vocab)

        print("Top 10")
        print(top10)

        print("Bottom 10")
        print(bottom10)

        filename = "./saved/bow_50words_{}_mindf5.csv".format(int(theta * 100))
        df_bow.to_csv(filename, index_label="PaperId")
        print("DF shape {} saved to {}".format(df_bow.shape, filename))

        print("\n---\n")
