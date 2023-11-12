from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sn


def print_columns(df):
    print("[ " + reduce(lambda x, y: x + ", " + y, df.columns, '')[2:] + " ]")


def get_overall_age(birth_dates):
    birth_years = []
    for i in birth_dates:
        birth_years.append(int(i.split("-")[0]))
    return sum(birth_years) / len(birth_years)


def select_features(df, target, key_features):
    available_columns = list(df.columns)
    available_columns.remove(target)
    features_values = pd.DataFrame(df, columns=available_columns)
    target_values = df["playoff"].values
    chi2(features_values, target_values)
    best_chi2_cols = SelectKBest(chi2, k=33)
    best_chi2_cols.fit(features_values, target_values)
    best_chi2_features = features_values.columns[best_chi2_cols.get_support()]

    key_predictors = set(best_chi2_features)
    key_predictors.update(key_features)

    df = df[list(key_predictors)]
    return df


def plot_pca(df):
    pca = PCA()
    pca.fit(df)  # X is your data
    explained_var_ratio = pca.explained_variance_ratio_
    plt.plot(range(1, len(explained_var_ratio) + 1), explained_var_ratio)
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.show()

    cum_var_ratio = np.cumsum(explained_var_ratio)
    plt.plot(range(1, len(cum_var_ratio) + 1), cum_var_ratio)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.show()


def feature_aggregation_pca(df, n_components, columns_to_keep):
    column_names = [f'PC{i + 1}' for i in range(n_components)]  # Create custom column names

    df_to_keep = pd.DataFrame(df[columns_to_keep])
    df.drop(columns_to_keep, axis=1, inplace=True)

    # Assuming 'X' is your data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)  # Choose the number of components you want
    x_pca = pca.fit_transform(x_scaled)

    df_result = pd.DataFrame(data=x_pca, columns=column_names)

    for col in columns_to_keep:
        df_result[col] = df_to_keep[col].reset_index(drop=True)

    return df_result


def plot_correlation(df, figsize=(30, 30), dpi=480):
    sn.heatmap(df.corr(), annot=True, fmt='.2f')
    plt.figure(figsize=figsize, dpi=dpi)
    plt.show()
