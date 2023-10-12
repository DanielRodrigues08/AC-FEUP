import pandas as pd

def nulls_by_column(df):
    return df.isnull().sum()

def filter_column_uniques(df,size=1):
    df_clean = df.copy()
    for col in df.columns:
        if len(pd.unique(df[col])) == size:
            df_clean.drop(col, axis=1, inplace=True)
    return df_clean