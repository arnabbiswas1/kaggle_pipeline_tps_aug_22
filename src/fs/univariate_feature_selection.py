"""Custom univariate feature selection wrapper on differenr univariate
feature selection models from skelarn

Sample Usage:

    ufs = UnivariateFeatureSelection(
        feature_sel_type=UnivarFeatureSelectorType.SELECT_K_BEST,
        n_features=10,
        scoring=ScoringType.F_CLASSIF
    )
    ufs.fit(X, y)
    X_transformed = X.loc[:, ufs.get_support()]


"""
from enum import Enum
from typing import Union

from sklearn.feature_selection import (SelectKBest, SelectPercentile, chi2,
                                       f_classif, f_regression,
                                       mutual_info_classif,
                                       mutual_info_regression)


__all__ = [
    "UnivarFeatureSelectorType",
    "ScoringType",
    "UnivariateFeatureSelection"
]


class UnivarFeatureSelectorType(Enum):
    SELECT_K_BEST = "SelectKBest"
    SELECT_PERCENTILE = "SelectPercentile"


class ScoringType(Enum):
    F_CLASSIF = f_classif
    CHI2_CLASSIF = chi2
    MUTUAL_INFO_CLASSIF = mutual_info_classif
    F_REGRESSION = f_regression
    MUTUAL_INFO_REGRESSION = mutual_info_regression


class UnivariateFeatureSelection:
    def __init__(
        self,
        feature_sel_type: UnivarFeatureSelectorType,
        n_features: Union[int, float],
        scoring: ScoringType,
    ):
        if feature_sel_type == UnivarFeatureSelectorType.SELECT_K_BEST:
            if not isinstance(n_features, int):
                raise ValueError(
                    "n_features is expected to be of type int for SelectKBest"
                )
            self.selector = SelectKBest(score_func=scoring, k=n_features)
        elif feature_sel_type == UnivarFeatureSelectorType.SELECT_PERCENTILE:
            if not isinstance(n_features, float):
                raise ValueError(
                    "n_features is expected to be of type float for SelectPercentile"
                )
            self.selector = SelectPercentile(
                score_func=scoring, percentile=int(n_features * 100)
            )
        else:
            raise ValueError("Proper Selection Type not selected")

    def fit(self, X, y):
        return self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def fit_transform(self, X, y):
        return self.selector.fit_transform(X, y)

    def get_support(self, indices=False):
        return self.selector.get_support(indices=False)


# if __name__ == "__main__":
#     ufs = UnivariateFeatureSelection(
#         feature_sel_type=UnivarFeatureSelectorType.SELECT_K_BEST,
#         n_features=10,
#         scoring=ScoringType.F_CLASSIF
#     )
#     ufs.fit(X, y)
#     X_transformed = X.loc[:, ufs.get_support()]
