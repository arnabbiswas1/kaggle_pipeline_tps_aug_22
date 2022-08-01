"""A library consisting of different feature selection mechanisms
"""

from sklearn.feature_selection import VarianceThreshold


__all__ = [
    "apply_variance_threshold"
]


def apply_variance_threshold(logger, df, threshold=0.1):
    """
    Removes features with very low variance using VarianceThreshold
    and returns DataFrame containing selected features
    """
    logger.info(
        f"Selecting features using VarianceThreshold with threshold {threshold}"
    )
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    # Return the DF with selected features
    # https://stackoverflow.com/a/41041230/406896
    df_sel = df.loc[:, selector.get_support()]
    return df_sel
