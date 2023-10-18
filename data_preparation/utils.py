import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sb


def nulls_values_by_column(df):
    null_counts = df.isnull().sum()
    plot = null_counts.plot(kind="bar", title="Number of null values per column")
    for i, v in enumerate(null_counts):
        plot.text(i, v, str(v), ha="center", va="bottom")
    return plot


def unique_values_by_column(df, threshold=0):
    counts = {col: len(df[col].unique()) for col in df.columns}
    counts_df = pd.DataFrame.from_dict(counts, orient="index", columns=["count"])

    ax = counts_df.plot(kind="bar", title="Number of Unique Values Per Column")

    [
        bar.set_color("green") if bar.get_height() > threshold else bar.set_color("red")
        for bar in ax.patches
    ]

    for i, v in enumerate(counts_df["count"]):
        ax.annotate(str(v), xy=(i, v), ha="center", va="bottom")

    ax.axhline(y=threshold, color="k", linestyle="--")

    return ax


def filter_column_uniques(df, size=1):
    df_clean = df.copy()
    for col in df.columns:
        if len(pd.unique(df[col])) == size:
            df_clean.drop(col, axis=1, inplace=True)
    return df_clean


def histogram_plot(df, max_zscore=3):
    numerical_columns = df.select_dtypes(include=["number"]).columns

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
        ax.axvline(
            x=df[column].mean() - std_dev, color="g", linestyle="--", label="std dev"
        )
        ax.axvline(x=df[column].mean() + std_dev, color="g", linestyle="--")
        ax.axvline(
            x=df[column].mean() - std_dev * max_zscore,
            color="r",
            linestyle="--",
            label="z-score",
        )
        ax.axvline(
            x=df[column].mean() + std_dev * max_zscore, color="r", linestyle="--"
        )
        ax.fill_betweenx(
            ax.get_ylim(),
            df[column].mean() - std_dev * max_zscore,
            df[column].mean() + std_dev * max_zscore,
            alpha=0.1,
            color="g",
        )
        ax.legend()

    # If the number of variables is odd, remove the empty subplot
    if num_columns % 2 != 0:
        fig.delaxes(axes[num_rows - 1, 1])

    plt.tight_layout()
    return plt


def calculate_bounds_iqr(col, factor=1.5):
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return lower_bound, upper_bound


def iqr_plot(df):
    numerical_columns = df.select_dtypes(include=["number"]).columns

    num_columns = len(numerical_columns)
    num_rows = (num_columns + 1) // 2

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 8))

    for i, column in enumerate(numerical_columns):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        sb.boxplot(x=df[column], orient="h", ax=ax)
        lower_bound, upper_bound = calculate_bounds_iqr(df[column])
        ax.fill_betweenx(
            ax.get_ylim(),
            lower_bound,
            upper_bound,
            alpha=0.1,
            color="g",
        )

    # If the number of variables is odd, remove the empty subplot
    if num_columns % 2 != 0:
        fig.delaxes(axes[num_rows - 1, 1])

    plt.tight_layout()
    return plt


def scatter_plot(df):
    # Define the number of columns per row
    columns_per_row = 3
    numerical_columns = df.select_dtypes(include=["number"]).columns

    # Calculate the total number of rows needed
    total_plots = len(numerical_columns) * (len(numerical_columns) - 1) // 2
    total_rows = (total_plots + columns_per_row - 1) // columns_per_row

    # Create the overall figure and axis objects
    fig, axs = plt.subplots(total_rows, columns_per_row, figsize=(15, 10))

    # Flatten the axis objects to make indexing easier
    axs = axs.flatten()

    # Iterate over the numerical columns combinations
    plot_index = 0
    for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            # Set the current axis for the plot
            ax = axs[plot_index]

            # Generate the scatter plot
            sb.scatterplot(
                data=df, x=numerical_columns[i], y=numerical_columns[j], ax=ax
            )

            # Set plot title, x-axis label, and y-axis label
            ax.set_title(
                f"Scatter Plot: {numerical_columns[i]} vs {numerical_columns[j]}"
            )
            ax.set_xlabel(numerical_columns[i])
            ax.set_ylabel(numerical_columns[j])

            # Move to the next axis
            plot_index += 1

    # Remove empty subplots if the total number of plots is not a multiple of columns_per_row
    if total_plots % columns_per_row != 0:
        for j in range(total_plots % columns_per_row, columns_per_row):
            fig.delaxes(axs[plot_index])

    # Adjust spacing between subplots
    fig.tight_layout()

    # Display the plots
    plt.show()


def filter_by_zscore(df, threshold=3, exclude=[]):
    eligible_columns = df.select_dtypes(include=["int", "float"]).columns
    eligible_columns = list(set(eligible_columns) - set(exclude))
    z_scores = np.abs(stats.zscore(df[eligible_columns]))
    return set(np.where(z_scores > threshold)[0])


def filter_by_iqr(df, factor=1.5, exclude=[]):
    eligible_columns = df.select_dtypes(include=["int", "float"]).columns
    eligible_columns = list(set(eligible_columns) - set(exclude))
    rows2drop = set()
    for column in eligible_columns:
        lower_bound, upper_bound = calculate_bounds_iqr(df[column], factor)
        rows2drop.update(
            set(df[(df[column] < lower_bound) | (df[column] > upper_bound)].index)
        )
    return rows2drop


def scatter_plot_mean_distance(df):
    # Define the number of columns per row
    columns_per_row = 3
    numerical_columns = df.select_dtypes(include=["number"]).columns

    # Calculate the total number of rows needed
    total_plots = len(numerical_columns) * (len(numerical_columns) - 1) // 2
    total_rows = (total_plots + columns_per_row - 1) // columns_per_row

    # Create the overall figure and axis objects
    fig, axs = plt.subplots(total_rows, columns_per_row, figsize=(15, 10))

    # Flatten the axis objects to make indexing easier
    axs = axs.flatten()

    # Iterate over the numerical columns combinations
    plot_index = 0
    for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            # Set the current axis for the plot
            ax = axs[plot_index]

            # Generate the scatter plot
            sb.scatterplot(
                data=df, x=numerical_columns[i], y=numerical_columns[j], ax=ax
            )

            # Set plot title, x-axis label, and y-axis label
            ax.set_title(
                f"Scatter Plot: {numerical_columns[i]} vs {numerical_columns[j]}"
            )
            ax.set_xlabel(numerical_columns[i])
            ax.set_ylabel(numerical_columns[j])

            # Move to the next axis
            plot_index += 1

    # Remove empty subplots if the total number of plots is not a multiple of columns_per_row
    if total_plots % columns_per_row != 0:
        for j in range(total_plots % columns_per_row, columns_per_row):
            fig.delaxes(axs[plot_index])

    # Adjust spacing between subplots
    fig.tight_layout()

    # Display the plots
    plt.show()