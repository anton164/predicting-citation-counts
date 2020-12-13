import pandas as pd


def load_df(filename):
    df = pd.read_csv(filename, index_col="PaperId")
    return df


def join_dfs(df1, df2):
    return df1.join(df2, on="PaperId")


def create_results_df(y_paper_ids, y_true, y_pred):
    """
    Create a results df to save model predictions:
    TrueClass, PredictedClass, Misclassified,
    MisclassifiedAsHigh, MisclassifiedAsLow
    """
    results_df = pd.DataFrame(
        list(zip(y_paper_ids, y_true, y_pred)),
        columns=["PaperId", "TrueClass", "PredictedClass"],
    ).set_index("PaperId")

    results_df["Misclassified"] = results_df.apply(
        lambda x: 1 if x["TrueClass"] != x["PredictedClass"] else 0, axis=1
    )
    results_df["MisclassifiedAsHigh"] = results_df.apply(
        lambda x: 1 if x["PredictedClass"] == 1 and x["Misclassified"] else 0, axis=1
    )
    results_df["MisclassifiedAsLow"] = results_df.apply(
        lambda x: 1 if x["PredictedClass"] == 0 and x["Misclassified"] else 0, axis=1
    )

    return results_df
