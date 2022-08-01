from typing import Tuple

# import matplotlib
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# matplotlib.style.use("dark_background")


__all__ = [
    "plot_desnity_train_test_overlapping",
    "plot_hist_train_test_overlapping",
    "plot_barh_train_test_side_by_side",
    "plot_line_train_test_overlapping",
    "plot_hist",
    "plot_point",
    "plot_vanilla_barh",
    "plot_barh",
    "plot_boxh",
    "plot_line",
    "plot_boxh_groupby",
    "plot_boxh_train_test_overlapping",
    "plot_hist_groupby",
    "save_feature_importance_as_fig",
    "save_permutation_importance_as_fig",
    "save_optuna_param_importance_as_fig",
    "save_rfecv_plot",
    "plot_seasonal_decomposition",
    "plot_seasonality",
    "plot_trend",
    "plot_ts_line_groupby",
    "plot_ts_point_groupby",
    "plot_ts_bar_groupby",
    "plot_multiple_seasonalities",
    "plot_confusion_matrix",
    "plot_acf_pacf_for_feature",
    "plot_acf_pacf_for_series",
    "plot_null_percentage_train_test_side_by_side",
    "plot_point_train_test_side_by_side_w_color_based_on_target"
]


def plot_desnity_train_test_overlapping(df_train, df_test, feature_name):
    """
    Plot density for a particular feature both for train and test.

    """
    df_train[feature_name].plot.density(
        figsize=(15, 5),
        label="train",
        alpha=0.4,
        color="red",
        title=f"Train vs Test {feature_name} distribution",
    )
    df_test[feature_name].plot.density(label="test", alpha=0.4, color="blue")
    plt.legend()
    plt.show()


def plot_hist_train_test_overlapping(
    df_train, df_test, feature_name, kind="hist", figsize=(10, 10), bins=100
):
    """
    Plot histogram for a particular feature both for train and test.

    kind : Type of the plot

    """
    df_train[feature_name].plot(
        kind=kind,
        figsize=figsize,
        label="train",
        bins=bins,
        alpha=0.4,
        color="red",
        title=f"Train vs Test {feature_name} distribution",
    )
    df_test[feature_name].plot(
        kind="hist",
        figsize=figsize,
        label="test",
        bins=bins,
        alpha=0.4,
        color="darkgreen",
    )
    plt.legend()
    plt.show()


def plot_barh_train_test_side_by_side(
    df_train, df_test, feature_name, normalize=True, sort_index=False
):
    """
    Plot histogram for a particular feature both for train and test.

    kind : Type of the plot

    """
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 8))

    if sort_index:
        ax1 = (
            df_train[feature_name]
            .value_counts(normalize=normalize, dropna=False)
            .sort_index()
            .plot(
                kind="barh",
                figsize=(15, 6),
                ax=ax1,
                grid=True,
                title=f"Bar plot for {feature_name} for train",
            )
        )

        ax2 = (
            df_test[feature_name]
            .value_counts(normalize=normalize, dropna=False)
            .sort_index()
            .plot(
                kind="barh",
                figsize=(15, 6),
                ax=ax2,
                grid=True,
                title=f"Bar plot for {feature_name} for test",
            )
        )
    else:
        ax1 = (
            df_train[feature_name]
            .value_counts(normalize=normalize, dropna=False)
            .sort_values()
            .plot(
                kind="barh",
                figsize=(15, 6),
                ax=ax1,
                grid=True,
                title=f"Bar plot for {feature_name} for train",
            )
        )

        ax2 = (
            df_test[feature_name]
            .value_counts(normalize=normalize, dropna=False)
            .sort_values()
            .plot(
                kind="barh",
                figsize=(15, 6),
                ax=ax2,
                grid=True,
                title=f"Bar plot for {feature_name} for test",
            )
        )
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    plt.legend()
    plt.show()


def plot_null_percentage_train_test_side_by_side(df_train, df_test, figsize=(20, 20)):
    """
    Plot histogram for a particular feature both for train and test.

    kind : Type of the plot

    """
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=figsize)

    null_train = (df_train.isna().sum().mul(100)) / len(df_train)
    null_test = (df_test.isna().sum().mul(100)) / len(df_test)

    null_train.plot(
        kind="barh", ax=ax1, grid=True, title="Percentage of null in train",
    )
    null_test.plot(
        kind="barh", ax=ax2, grid=True, title="Percentage of null in test",
    )
    plt.show()


def plot_line_train_test_overlapping(df_train, df_test, feature_name, figsize=(10, 5)):
    """
    Plot line for a particular feature both for train and test
    """
    df_train[feature_name].plot(
        kind="line",
        figsize=figsize,
        label="train",
        alpha=0.4,
        title=f"Train vs Test {feature_name} distribution",
    )
    df_test[feature_name].plot(kind="line", label="test", alpha=0.4)
    plt.ylabel(f"Value of {feature_name}")
    plt.legend()
    plt.show()


def plot_line(df, feature_name, figsize=(10, 5)):
    """
    Plot line for a particular feature for the DF
    """
    df[feature_name].plot(
        kind="line",
        figsize=figsize,
        label="train",
        alpha=0.4,
        title=f"Line plot for {feature_name} distribution",
    )
    plt.ylabel(f"Value of {feature_name}")
    plt.legend()
    plt.show()


def plot_point(df, feature_name, figsize=(10, 5)):
    """
    Plot line for a particular feature for the DF
    """
    df[feature_name].plot(
        kind="line",
        style=".",
        figsize=figsize,
        alpha=0.4,
        title=f"Plot for {feature_name} distribution",
    )
    plt.ylabel(f"Value of {feature_name}")
    plt.legend()
    plt.show()


def plot_hist(df, feature_name, kind="hist", bins=100, log=True):
    """
    Plot either for train or test
    """
    if log:
        df[feature_name].apply(np.log1p).plot(
            kind="hist",
            bins=bins,
            figsize=(15, 5),
            title=f"Distribution of log1p[{feature_name}]",
        )
    else:
        df[feature_name].plot(
            kind="hist",
            bins=bins,
            figsize=(15, 5),
            title=f"Distribution of {feature_name}",
        )
    plt.show()


def plot_vanilla_barh(df, index_name, feature_name, figsize=(15, 5)):
    """
    Plot barh for a particular feature directly without calculating the value_counts
    """
    ax = df.set_index(index_name)[feature_name].plot(
        kind="barh", title=f"Distribtion of {feature_name}", figsize=figsize,
    )
    ax.invert_yaxis()
    plt.legend()
    plt.show()


def plot_barh(
    df, feature_name, normalize=True, kind="barh", figsize=(15, 5), sort_index=False
):
    """
    Plot barh for a particular feature both for train and test.

    kind : Type of the plot

    """
    if sort_index:
        ax = (
            df[feature_name]
            .value_counts(normalize=normalize, dropna=False)
            .sort_index()
            .plot(
                kind=kind,
                figsize=figsize,
                grid=True,
                title=f"Bar plot for {feature_name}",
            )
        )
    else:
        ax = (
            df[feature_name]
            .value_counts(normalize=normalize, dropna=False)
            .sort_values(ascending=True)
            .plot(
                kind=kind,
                figsize=figsize,
                grid=True,
                title=f"Bar plot for {feature_name}",
            )
        )
    ax.invert_yaxis()
    plt.legend()
    plt.show()


def plot_boxh_train_test_overlapping(
    train_df, test_df, feature_name, kind="box", log=False, figsize=(10, 4)
):
    """
    Box plot train and test
    """
    fig, ((ax1, ax2)) = plt.subplots(2, 1, sharex=True, figsize=figsize)
    if log:
        (
            train_df[feature_name]
            .apply(np.log1p)
            .plot(
                kind="box",
                vert=False,
                ax=ax1,
                label="train",
                title=f"Distribution of log1p[{feature_name}]",
            )
        )
        (
            test_df[feature_name]
            .apply(np.log1p)
            .plot(kind="box", vert=False, label="test", ax=ax2)
        )
    else:
        ax1 = train_df[feature_name].plot(
            kind="box",
            vert=False,
            ax=ax1,
            subplots=False,
            label="train",
            title=f"Distribution of {feature_name}",
        )
        ax2 = test_df[feature_name].plot(kind="box", vert=False, label="test", ax=ax2)
    plt.show()


def plot_point_train_test_side_by_side_w_color_based_on_target(train_df, test_df, feature_name, target, figsize=(20, 4)):
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    N = 5000
    train_df = train_df[0: N]
    test_df = test_df[0: N]
    train_df[train_df[target] == 0][feature_name].plot(
        style=".",
        alpha=0.3,
        ax=ax1,
        color="blue",
    )
    train_df[train_df[target] == 1][feature_name].plot(
        style=".",
        alpha=0.2,
        ax=ax1,
        color="orange",
    )
    test_df[feature_name].plot(
        style=".",
        alpha=0.2,
        ax=ax2,
        color="green",
    )
    ax1.set_title(f"{feature_name} train [First {N} rows] (blue=no-claim, orange=claim)")
    ax2.set_title(f"{feature_name} test [First {N} rows] (green)")
    plt.ylabel(f"Value of {feature_name}")
    plt.show()


def plot_boxh(df, feature_name, kind="box", log=True):
    """
    Box plot either for train or test
    """
    if log:
        df[feature_name].apply(np.log1p).plot(
            kind="box",
            vert=False,
            figsize=(10, 6),
            title=f"Distribution of log1p[{feature_name}]",
        )
    else:
        df[feature_name].plot(
            kind="box",
            vert=False,
            figsize=(10, 6),
            title=f"Distribution of {feature_name}",
        )
    plt.show()


def plot_boxh_groupby(df, feature_name, by):
    """
    Box plot with groupby feature
    """
    df.boxplot(column=feature_name, by=by, vert=False, figsize=(10, 6))
    plt.title(f"Distribution of {feature_name} by {by}")
    plt.show()


def plot_hist_groupby(df, feature_name, by, bins=100, figsize=(15, 5)):
    """
    Box plot with groupby feature
    """
    df.hist(column=feature_name, by=by, figsize=figsize, bins=100, legend=False)
    plt.suptitle(f"Distribution of {feature_name} by {by}")
    plt.show()


def save_feature_importance_as_fig(best_features_df, dir_name, file_name):
    plt.figure(figsize=(16, 12))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features_df.sort_values(by="importance", ascending=False),
    )
    plt.title("Features (avg over folds)")
    plt.savefig(f"{dir_name}/{file_name}")


def save_permutation_importance_as_fig(best_features_df, dir_name, file_name):
    plt.figure(figsize=(16, 12))
    sns.barplot(
        x="weight",
        y="feature",
        data=best_features_df.sort_values(by="weight", ascending=False),
    )
    plt.title("Permutation Importance (avg over folds)")
    plt.savefig(f"{dir_name}/{file_name}")


def save_optuna_param_importance_as_fig(params_df, dir_name, file_name):
    plt.figure(figsize=(16, 12))
    sns.barplot(
        x="importance",
        y="param_name",
        data=params_df.sort_values(by="importance", ascending=False),
    )
    plt.title("Importance of hyperparameters")
    plt.savefig(f"{dir_name}/{file_name}")


def save_rfecv_plot(rfecv, dir_name, file_name):
    plt.figure(figsize=(14, 8))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.title("CV Score vs Features Selected (REFCV)")
    plt.savefig(f"{dir_name}/{file_name}")


def plot_seasonal_decomposition(
    df, feature, freq, freq_type="daily", model="additive", figsize=(20, 10)
):
    plt.rcParams["figure.figsize"] = figsize
    decomposition = tsa.seasonal_decompose(df[feature], model=model, period=freq)
    decomposition.plot()
    plt.title(f"{model} {freq_type} seasonal decomposition of {feature}")
    plt.show()


def plot_seasonality(
    df, feature, freq, freq_type="daily", model="additive", figsize=(20, 10)
):
    plt.rcParams["figure.figsize"] = figsize
    decomposition = tsa.seasonal_decompose(df[feature], model=model, period=freq)
    decomposition.seasonal.plot(color="blue", linewidth=0.5)
    plt.title(f"{model} {freq_type} seasonality of {feature}")
    plt.show()

 
def plot_trend(
    df, feature, freq, freq_type="daily", model="additive", figsize=(20, 10)
):
    plt.rcParams["figure.figsize"] = figsize
    decomposition = tsa.seasonal_decompose(df[feature], model=model, period=freq)
    decomposition.trend.plot(color="blue", linewidth=0.5)
    plt.title(f"{model} {freq_type} seasonality of {feature}")
    plt.show()


def get_colors(color_map_name: str = "Set1") -> Tuple:
    """
    https://matplotlib.org/stable/gallery/color/colormap_reference.html
    # type: matplotlib.colors.ListedColormap
    """
    cmap = get_cmap(color_map_name)
    return cmap.colors


def plot_ts_line_groupby(
    df, ts_index_feature, groupby_feature, value_feature, figsize=(15, 8),
):
    colors = get_colors()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_prop_cycle(color=colors)
    for label, df in df.groupby(groupby_feature):
        df.set_index(ts_index_feature)[value_feature].plot(
            kind="line", alpha=0.3, ax=ax, linewidth=0.1,
        )
    plt.title(f"Time Series Plot for {value_feature}")
    plt.xlabel("Time")
    plt.ylabel(value_feature)
    # plt.legend()
    plt.show()


def plot_ts_point_groupby(
    df, ts_index_feature, groupby_feature, value_feature, figsize=(15, 8),
):
    colors = get_colors()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_prop_cycle(color=colors)
    for label, df in df.groupby(groupby_feature):
        df.set_index(ts_index_feature)[value_feature].plot(
            style=".", kind="line", alpha=0.3, ax=ax, linewidth=0.1
        )
    plt.title(f"Time Series Plot for {value_feature}")
    plt.xlabel("Time")
    plt.ylabel(value_feature)
    # plt.legend(False)
    plt.show()


def plot_ts_bar_groupby(
    df,
    ts_index_feature,
    groupby_feature,
    value_feature,
    title,
    xlabel,
    ylabel,
    figsize=(15, 8),
):
    fig, ax = plt.subplots(figsize=figsize)
    for label, df in df.groupby(groupby_feature):
        df.set_index(ts_index_feature)[value_feature].plot(
            kind="bar", alpha=0.05, ax=ax, linewidth=0.5
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_multiple_seasonalities(df, feature_name, figsize=(20, 6)):
    period_names = ["daily", "weekly", "monthly", "quarterly"]
    periods = [24, 24 * 7, 24 * 30, 24 * 90]

    for name, period in zip(period_names, periods):
        if "date_time" in df.columns:
            plot_seasonality(
                df.set_index("date_time")[0: period * 3],
                feature=feature_name,
                freq=period,
                freq_type=name,
                figsize=figsize,
            )
        else:
            plot_seasonality(
                df[0: period * 3],
                feature=feature_name,
                freq=period,
                freq_type=name,
                figsize=figsize,
            )


def plot_confusion_matrix(cm_array, labels, figsize):
    """
    cm_array: confusion matrix generated by sklearn's confusion_matrix()
    labels = List of classes in order
    """
    df_cm = pd.DataFrame(
        cm_array, index=[i for i in labels], columns=[i for i in labels]
    )
    plt.figure(figsize=figsize)
    sns.heatmap(df_cm, annot=True, fmt="d")
    plt.xlabel("Actual")
    plt.ylabel("Prediction")
    plt.show()


def plot_acf_pacf_for_feature(df, feature_name, lags=50, figsize=(10, 4)):
    """
    Plot ACF and PACF side by side
    """
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=figsize)
    plot_acf(df[feature_name], lags=lags, ax=ax1, title=f"ACF for {feature_name}")
    plot_pacf(df[feature_name], lags=lags, ax=ax2, title=f"PACF for {feature_name}")
    plt.show()


def plot_acf_pacf_for_series(ser, lags=50, title="", figsize=(10, 4)):
    """
    Plot ACF and PACF side by side
    """
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=figsize)
    plot_acf(ser, ax=ax1, lags=lags, title=f"ACF for {title}")
    plot_pacf(ser, ax=ax2, lags=lags, title=f"PACF for {title}")
    plt.show()
