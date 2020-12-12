import pandas as pd

def load_df(filename):
    df = pd.read_csv(filename, index_col="PaperId") 
    return df

def join_dfs(df1, df2):
    return df1.join(df2, on='PaperId')

def create_error_analysis_df(y_paper_ids, y_true, y_pred, full_df):
    """ 
        Creates an error analysis df with cols from the full df and: 
        TrueClass, PredictedClass, Misclassified,
        MisclassifiedAsHigh, MisclassifiedAsLow 
    """
    error_analysis_df = pd.DataFrame(
        list(zip(y_index, y_true, y_pred)),
        columns=["PaperId", "TrueClass", "PredictedClass"]
    ).set_index("PaperId")

    error_analysis_df = join_dfs(error_analysis_df, full_df)
    error_analysis_df["Misclassified"] = error_analysis_df.apply(lambda x: 1 if x["TrueClass"] != x["PredictedClass"] else 0, axis=1)
    error_analysis_df["MisclassifiedAsHigh"] = error_analysis_df.apply(
        lambda x: 1 if x["PredictedClass"] == 1 and x["Misclassified"] else 0, 
        axis=1
    )
    error_analysis_df["MisclassifiedAsLow"] = error_analysis_df.apply(
        lambda x: 1 if x["PredictedClass"] == 0 and x["Misclassified"] else 0, 
        axis=1
    )

    return error_analysis_df