import pandas as pd
import numpy as np


__all__ = [
    "create_date_features",
    "create_us_season",
    "create_part_of_day",
    "fill_with_gauss",
    "fill_with_po3",
    "fill_with_lin",
    "fill_with_mix",
    "find_missing_dates",
    "get_start_end_date",
    "get_first_date_string",
    "get_last_date_string",
]


def create_part_of_day(source_df, target_df, feature_name):
    hour_bins = [0, 4, 8, 12, 16, 20, 24]
    labels = ["late_night", "early_morning", "morning", "noon", "evening", "night"]
    target_df.loc[:, "part_of_day"] = pd.cut(
        source_df.loc[:, feature_name].dt.hour, bins=hour_bins, labels=labels, include_lowest=True
    )
    return target_df


def create_us_season(source_df, target_df, feature_name):
    """
    Winter: December - February (12. 1, 2)
    Spring: March - May (3. 4, 5)
    Summer: June - August [6, 7, 8]
    Fall: September - November (9, 10, 11)
    """
    month_to_season_map = {
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "fall",
        10: "fall",
        11: "fall",
        12: "winter",
    }
    target_df.loc[:, "us_season"] = source_df.loc[:, feature_name].dt.month.map(
        month_to_season_map
    )
    return target_df


def create_date_features(source_df, target_df, feature_name):
    """
    Create new features related to dates

    source_df : DataFrame consisting of the timestamp related feature
    target_df : DataFrame where new features will be added
    feature_name : Name of the feature of date type which needs to be decomposed.
    """
    target_df.loc[:, "year"] = source_df.loc[:, feature_name].dt.year.astype(np.int16)
    target_df.loc[:, "month"] = source_df.loc[:, feature_name].dt.month.astype(np.int8)
    target_df.loc[:, "quarter"] = source_df.loc[:, feature_name].dt.quarter.astype(
        np.int8
    )
    target_df.loc[:, "weekofyear"] = (
        source_df.loc[:, feature_name].dt.isocalendar().week.astype(np.int8)
    )

    target_df.loc[:, "hour"] = source_df.loc[:, feature_name].dt.hour.astype(np.int8)
    # target_df.loc[:, 'minute'] = source_df.loc[:, feature_name].dt.minute.astype('uint32')
    # target_df.loc[:, 'second'] = source_df.loc[:, feature_name].dt.second.astype('uint32')

    target_df.loc[:, "day"] = source_df.loc[:, feature_name].dt.day.astype(np.int8)
    target_df.loc[:, "day_name"] = source_df.loc[:, feature_name].dt.day_name()
    target_df.loc[:, "dayofweek"] = source_df.loc[:, feature_name].dt.dayofweek.astype(
        np.int8
    )

    target_df.loc[:, "day_type"] = np.where(
        source_df.loc[:, feature_name].dt.dayofweek < 5, "week_day", "week_end"
    )
    target_df.loc[:, "dayofyear"] = source_df.loc[:, feature_name].dt.dayofyear.astype(
        np.int8
    )
    target_df.loc[:, "is_month_start"] = source_df.loc[
        :, feature_name
    ].dt.is_month_start
    target_df.loc[:, "is_month_end"] = source_df.loc[:, feature_name].dt.is_month_end
    target_df.loc[:, "is_quarter_start"] = source_df.loc[
        :, feature_name
    ].dt.is_quarter_start
    target_df.loc[:, "is_quarter_end"] = source_df.loc[
        :, feature_name
    ].dt.is_quarter_end
    target_df.loc[:, "is_year_start"] = source_df.loc[:, feature_name].dt.is_year_start
    target_df.loc[:, "is_year_end"] = source_df.loc[:, feature_name].dt.is_year_end

    # This is of type object
    target_df.loc[:, "month_year"] = source_df.loc[:, feature_name].dt.to_period("M")

    return target_df


def fill_with_gauss(ser, w=12):
    """
    Fill missing values in a time series data using gaussian
    """
    return ser.fillna(
        ser.rolling(window=w, win_type="gaussian", center=True, min_periods=1).mean(
            std=2
        )
    )


def fill_with_po3(ser):
    """
    Fill missing values in a time series data using interpolation (polynomial, order 3)
    """
    return ser.fillna(ser.interpolate(method="polynomial", order=3, limit_direction='both'))
    # assert df.count().min() >= len(df) - 1
    # fill the first item with second item
    # return df.fillna(df.iloc[1])
    # return ser


def fill_with_lin(ser):
    """
    Fill missing values in a time series data using interpolation (linear)
    """
    return ser.fillna(ser.interpolate(method="linear", limit_direction='both'))
    # assert df.count().min() >= len(df) - 1
    # fill the first item with second item
    # return df.fillna(df.iloc[1])


def fill_with_mix(ser):
    """
    Fill missing values in a time series data using interpolation (linear + polynomial)
    """
    ser = (
        ser.fillna(ser.interpolate(method="linear", limit_direction="both"))
        + ser.fillna(
            ser.interpolate(method="polynomial", order=3, limit_direction="both")
        )
    ) * 0.5
    return ser
    # assert df.count().min() >= len(df) - 1
    # fill the first item with second item
    # return df.fillna(df.iloc[1])


def find_missing_dates(date_sr, start_date, end_date):
    """
    Returns the dates which are missing in the series
    date_sr between the start_date and end_date

    date_sr: Series consisting of date
    start_date: Start date in String format
    end_date: End date in String format
    """
    return pd.date_range(start=start_date, end=end_date, freq="H").difference(date_sr)


def get_start_end_date(df, feature, format="%Y-%m-%d  %H:%M:%S"):
    """
    Returns the start and end date of a time series

    Args
        df: DataFrame consisting of the time series
        feature: Name of the time series feature. The column should be of type np.datetime64
        format: String format in which start and end date will be returned
    """
    start_date = df[feature].min().strftime(format="%Y-%m-%d  %H:%M:%S")
    end_date = df[feature].max().strftime(format="%Y-%m-%d  %H:%M:%S")
    return start_date, end_date


def get_first_date_string(date_sr, date_format="%Y-%m-%d"):
    """
    Returns the first date of the series date_sr

    date_sr: Series consisting of date
    date_format: Format to be used for converting date into String
    """
    return _get_boundary_date_string(date_sr, boundary="first", date_format="%Y-%m-%d")


def get_last_date_string(date_sr, date_format="%Y-%m-%d"):
    """
    Returns the last date of the series date_sr

    date_sr: Series consisting of date
    date_format: Format to be used for converting date into String
    """
    return _get_boundary_date_string(date_sr, boundary="last", date_format="%Y-%m-%d")


def _get_boundary_date_string(date_sr, boundary, date_format="%Y-%m-%d"):
    """
    Returns the first or last date of the series date_sr based on the
    value passed in boundary.

    date_sr: Series consisting of date
    boundary: Allowed values are 'first' or 'last'
    date_format: Format to be used for converting date into String
    """
    return date_sr.describe().loc[boundary].strftime(format="%Y-%m-%d")
