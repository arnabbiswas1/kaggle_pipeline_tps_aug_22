import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


__all__ = [
    "check_null",
    "check_duplicate",
    "count_unique_values",
    "do_value_counts",
    "check_id_column",
    "check_index",
    "get_feature_names",
    "check_value_counts_across_train_test",
    "get_freq_encoding_feature_names",
    "get_bin_feature_names",
    "get_power_feature_names",
    "get_row_wise_stat_feature_names",
    "get_cat_interaction_features",
    "get_features_with_no_variance",
    "check_if_floats_are_int",
    "create_summary_df",
    "generate_unique_count_summary",
    "calculate_skew_summary",
    "get_columns_with_null_values",
]


def check_null(df):
    """
    Calculate the percentage of null (missing) values for each column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: A pandas Series containing the percentage of null values for each column.

    Example Usage:
        null_percentages = check_null(data_df)
        print(null_percentages)
    """
    return df.isna().sum() * 100 / len(df)


def check_duplicate(df, subset):
    """
    Check for duplicate rows in a DataFrame and return the count of duplicate rows.

    Parameters:
        df (pd.DataFrame): The input DataFrame to check for duplicates.
        subset (list or None): A list of column names to consider when checking for duplicates.
                              If None, all columns are considered. Default is None.

    Returns:
        int: The count of duplicate rows in the DataFrame.

    Example Usage:
        # Check for duplicate rows in the entire DataFrame
        total_duplicates = check_duplicate(data_df)
        print(f"Total duplicate rows: {total_duplicates}")

        # Check for duplicate rows based on a subset of columns
        subset_duplicates = check_duplicate(data_df, subset=['column1', 'column2'])
        print(f"Subset duplicate rows: {subset_duplicates}")
    """
    if subset is not None:
        return df.duplicated(subset=subset, keep=False).sum()
    else:
        return df.duplicated(keep=False).sum()


def count_unique_values(df, feature_name):
    return df[feature_name].nunique()


def do_value_counts(df, feature_name):
    return (
        df[feature_name]
        .value_counts(normalize=True, dropna=False)
        .sort_values(ascending=False)
        * 100
    )


def check_index(df, data_set_name):
    """
    Check the properties of the DataFrame's index, including continuity and monotonicity, and visualize it.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        data_set_name (str): A descriptive name or label for the dataset used in the title of the plot.

    Returns:
        None

    Example Usage:
        # Check and visualize the index of a DataFrame
        check_index(data_df, "My Dataset")

    Description:
        This function examines the properties of the index of the input DataFrame 'df'
        and provides information about its continuity and monotonicity. It also generates
        a plot to visualize the index values.

        Continuity:
        - The function checks if the index is continuous, meaning that it does not contain
          any missing values or gaps in its sequence.

        Monotonicity:
        - The function checks if the index is monotonically increasing, indicating that
          the values in the index are strictly increasing from left to right.

        Visualization:
        - The function generates a plot to visualize the index values, providing a visual
          representation of the index's pattern.
    """
    print(f"Is the index monotonic : {df.index.is_monotonic}")
    # Plot the column
    pd.Series(df.index).plot(title=data_set_name)
    plt.show()


def check_id_column(df, column_name, data_set_name):
    """
    Check the properties of a specific identifier column in the DataFrame, including continuity
    and monotonicity, and visualize it.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the identifier column to check.
        data_set_name (str): A descriptive name or label for the dataset used in the title of the plot.

    Returns:
        None

    Example Usage:
        # Check and visualize the 'feature_3' column in a DataFrame
        check_id_column(data_df, 'feature_3', "My Dataset")

    Description:
        This function examines the properties of a specific identifier column ('column_name')
        in the input DataFrame 'df' and provides information about its continuity and monotonicity.
        It also generates a plot to visualize the values in the specified identifier column.

        Continuity:
        - The function checks if the specified identifier column is continuous, meaning that it
          does not contain any missing values or gaps in its sequence.

        Monotonicity:
        - The function checks if the specified identifier column is monotonically increasing,
          indicating that the values in the column are strictly increasing from top to bottom.

        Visualization:
        - The function generates a plot to visualize the values in the specified identifier column,
          providing a visual representation of the column's pattern.
    """
    print(f"Is the {column_name} monotonic : {df[column_name].is_monotonic}")
    # Plot the column
    df[column_name].plot(title=data_set_name)
    plt.show()


def get_feature_names(df, feature_name_substring):
    """
    Returns the list of features with name matching 'feature_name_substring'
    """
    return [
        col_name
        for col_name in df.columns
        if col_name.find(feature_name_substring) != -1
    ]


def check_value_counts_across_train_test(
    train_df, test_df, feature_name, normalize=True
):
    """
    Create a DF consisting of value_counts of a particular feature for
    train and test
    """
    train_counts = (
        train_df[feature_name]
        .sort_index()
        .value_counts(normalize=normalize, dropna=True)
        * 100
    )
    test_counts = (
        test_df[feature_name]
        .sort_index()
        .value_counts(normalize=normalize, dropna=True)
        * 100
    )
    count_df = pd.concat([train_counts, test_counts], axis=1).reset_index(drop=True)
    count_df.columns = [feature_name, "train", "test"]
    return count_df


def get_freq_encoding_feature_names(df):
    return get_feature_names(df, "freq")


def get_power_feature_names(df):
    power_features = []
    power_feature_keys = ["_square", "_cube", "_fourth", "_cp", "_cnp"]
    for name in df.columns:
        for key in power_feature_keys:
            if key in name:
                power_features.append(name)
    return power_features


def get_row_wise_stat_feature_names():
    return [
        "max",
        "min",
        "sum",
        "mean",
        "std",
        "skew",
        "kurt",
        "med",
        "ptp",
        "percentile_10",
        "percentile_60",
        "percentile_90",
        "quantile_5",
        "quantile_95",
        "quantile_99",
    ]


def get_bin_feature_names(df, bin_size=10):
    return get_feature_names(df, f"cut_bin_{bin_size}")


def get_cat_interaction_features():
    return ["f1_f86", "f1_f55",	"f1_f27", "f86_f55", "f86_f27", "f55_f27"]


def get_features_with_no_variance(df):
    return df.columns[df.nunique() <= 1]


def check_if_floats_are_int(df):
    """
    Returns name of the features which are of type float
    but actually contains whole numbers
    """
    int_features = []
    for column in df.columns:
        if np.all(np.mod(df[column], 1) == 0):
            print(f"Feature {column} does not have any decimals")
            int_features.append(column)
        else:
            print(f"Feature {column} have decimals")
    return int_features
