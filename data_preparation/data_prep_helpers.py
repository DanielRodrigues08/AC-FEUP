import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sb

def nulls_values_by_column(df):
    null_counts = df.isnull().sum()
    plot = null_counts.plot(kind='bar', title='Number of null values per column')
    for i, v in enumerate(null_counts):
        plot.text(i, v, str(v), ha='center', va='bottom')
    return plot


def unique_values_by_column(df, threshold=0):
    counts = {col: len(df[col].unique()) for col in df.columns}
    counts_df = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])

    ax = counts_df.plot(kind='bar', title='Number of Unique Values Per Column')

    [bar.set_color('green') if bar.get_height() > threshold else bar.set_color('red') for bar in ax.patches]

    for i, v in enumerate(counts_df['count']):
        ax.annotate(str(v), xy=(i, v), ha='center', va='bottom')

    ax.axhline(y=threshold, color='k', linestyle='--')

    return ax


def filter_column_uniques(df, size=1):
    df_clean = df.copy()
    for col in df.columns:
        if len(pd.unique(df[col])) == size:
            df_clean.drop(col, axis=1, inplace=True)
    return df_clean


def histogram_plot(df, max_zscore=3):
    numerical_columns = df.select_dtypes(include=['number']).columns

    num_columns = len(numerical_columns)
    num_rows = (num_columns + 1) // 2

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 8))

    for i, column in enumerate(numerical_columns):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        sb.histplot(data=df, x=column, kde=True, ax=ax)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        # Calculate and display the standard deviation
        std_dev = df[column].std()
        ax.axvline(x=df[column].mean() - std_dev, color='g', linestyle='--', label='std dev')
        ax.axvline(x=df[column].mean() + std_dev, color='g', linestyle='--')
        ax.axvline(x=df[column].mean() - std_dev * max_zscore, color='r', linestyle='--', label='z-score')
        ax.axvline(x=df[column].mean() + std_dev * max_zscore, color='r', linestyle='--')
        ax.fill_betweenx(ax.get_ylim(), df[column].mean() - std_dev * max_zscore, df[column].mean() + std_dev * max_zscore, alpha=0.1, color='g')
        ax.legend()

    # If the number of variables is odd, remove the empty subplot
    if num_columns % 2 != 0:
        fig.delaxes(axes[num_rows-1, 1])

    plt.tight_layout()
    return plt

def filter_by_zscore(df, threshold=3):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=['int', 'float'])))
    df_clean = df.drop(np.where(z_scores > threshold)[0])
    return df_clean
