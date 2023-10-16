import pandas as pd
import matplotlib.pyplot as plt


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

def filter_column_uniques(df,size=1):
    df_clean = df.copy()
    for col in df.columns:
        if len(pd.unique(df[col])) == size:
            df_clean.drop(col, axis=1, inplace=True)
    return df_clean


def histogram_plot(df, col, bins=10):
    df[col].hist(bins=bins)
    plt.title(col)
    plt.show()