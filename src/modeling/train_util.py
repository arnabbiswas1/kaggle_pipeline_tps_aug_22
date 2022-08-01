import gc

import eli5
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoost, Pool
from IPython.display import display
from sklearn import metrics
from sklearn.metrics import (
    make_scorer,
    roc_auc_score,
    mean_squared_error,
    f1_score,
)

import src.common.com_util as util


__all__ = [
    "cat_train_validate_on_holdout",
    "xgb_train_validate_on_holdout",
    "lgb_train_validate_on_holdout",
    "xgb_train_validate_on_cv",
    "lgb_train_validate_on_cv",
    "cat_train_validate_on_cv",
    "sklearn_train_validate_on_cv",
    "lgb_train_perm_importance_on_cv",
    "evaluate_macroF1_xgb",
    "evaluate_macroF1_lgb",
    "_calculate_perf_metric",
    "f1_score_weighted",
    "evaluate_macroF1_lgb_sklearn_api",
    "_get_X_Y_from_CV",
    "lgb_train_validate_on_cv_mean_encoding",
]


def evaluate_macroF1_lgb(y_hat, data):
    """
    Custom F1 Score to be used for multiclass classification using lightgbm.
    This function should be passed as a value to the parameter feval.

    weighted average takes care of imbalance

    https://stackoverflow.com/questions/57222667/light-gbm-early-stopping-does-not-work-for-custom-metric
    https://stackoverflow.com/questions/52420409/lightgbm-manual-scoring-function-f1-score
    https://stackoverflow.com/questions/51139150/how-to-write-custom-f1-score-metric-in-light-gbm-python-in-multiclass-classifica
    """
    y = data.get_label()
    y_hat = y_hat.reshape(-1, len(np.unique(y))).argmax(axis=1)
    f1 = f1_score(y_true=y, y_pred=y_hat, average="weighted")
    return ("weightedF1", f1, True)


def evaluate_macroF1_lgb_sklearn_api(y, y_hat):
    """
    Custom F1 Score to be used for multiclass classification using lightgbm.
    This function should be passed as a value to the parameter eval_metric for 
    the LGBM sklearn API.

    weighted average takes care of imbalance

    https://github.com/Microsoft/LightGBM/issues/1453

    """
    y_hat = y_hat.reshape(-1, len(np.unique(y))).argmax(axis=1)
    f1 = f1_score(y_true=y, y_pred=y_hat, average="weighted")
    return ("weightedF1", f1, True)


def evaluate_macroF1_xgb(y_hat, data):
    """
    Custom F1 Score to be used for multiclass classification using xgboost.
    This function should be passed as a value to the parameter feval.

    weighted average takes care of imbalance

    https://stackoverflow.com/questions/51587535/custom-evaluation-function-based-on-f1-for-use-in-xgboost-python-api
    https://www.kaggle.com/c/expedia-hotel-recommendations/discussion/21439
    """
    y = data.get_label()
    y_hat = y_hat.reshape(-1, len(np.unique(y))).argmax(axis=1)
    f1 = f1_score(y_true=y, y_pred=y_hat, average="weighted")
    return ("weightedF1", f1)


def f1_score_weighted(y, y_hat):
    """
    It's assumed that y_hat consists of a two dimensional array.
    Each array in the first dimension has probabilities for all the
    classes, i.e. if there are 43 classes and 1000 rows of data, the y_hat
    has a dimension (1000, 43)
    """
    y_hat = y_hat.reshape(-1, len(np.unique(y))).argmax(axis=1)
    return f1_score(y_true=y, y_pred=y_hat, average="weighted")


def roc_auc(y, y_hat):
    return roc_auc_score(y, y_hat)


def log_loss(y, y_hat):
    return metrics.log_loss(y_true=y, y_pred=y_hat)


def rmse(y, y_hat):
    return np.sqrt(mean_squared_error(y, y_hat))


def rmsle(y, y_hat):
    return np.sqrt(np.mean(np.power(np.log1p(y_hat) - np.log1p(y), 2)))


def precision_weighted(y, y_hat):
    return metrics.precision_score(y_true=y, y_pred=y_hat, average="weighted")


def recall_weighted(y, y_hat):
    return metrics.recall_score(y_true=y, y_pred=y_hat, average="weighted")


def _calculate_perf_metric(metric_name, y, y_hat):
    """Returns the performance metrics

       Args:
           y: Real value
           y_hat: predicted value

       Returns:
           Metrics computed
    """
    if metric_name == "rmse":
        return rmse(y, y_hat)
    elif metric_name == "rmsle":
        return rmsle(y, y_hat)
    elif metric_name == "roc_auc":
        return roc_auc(y, y_hat)
    elif metric_name == "log_loss":
        return log_loss(y, y_hat)
    elif metric_name == "f1_score_weighted":
        return f1_score_weighted(y, y_hat)
    elif metric_name == "precision_weighted":
        return precision_weighted(y, y_hat)
    elif metric_name == "recall_weighted":
        return recall_weighted(y, y_hat)
    else:
        raise ValueError(
            "Invalid value for metric_name. Only rmse, rmsle, roc_auc, log_loss allowed"
        )


def _get_scorer(metric_name):
    if metric_name == "roc_auc":
        return metrics.roc_auc_score
    elif metric_name == "log_loss":
        return metrics.log_loss
    else:
        raise ValueError(
            "Invalid value for metric_name. Only rmse, rmsle, roc_auc, log_loss allowed"
        )


def _get_random_seeds(i):
    """
    returns 10 seeds
    """
    seed_list = [42, 103, 13, 31, 17, 23, 46, 57, 83, 93]
    return seed_list[i - 1]


def _get_x_y_from_data(logger, df, predictors, target):
    """Returns X & Y from a DataFrame"""
    if df is not None:
        df_X = df[predictors]
        df_Y = df[target]
    return df_X, df_Y


def _get_x_y_from_training_validation(logger, training, validation, predictors, target):
    """Returns X & Y for training & validation data"""
    if training is not None:
        training_X, training_Y = _get_x_y_from_data(
            logger, training, predictors, target
        )
    if validation is not None:
        validation_X, validation_Y = _get_x_y_from_data(
            logger, validation, predictors, target
        )
    return training_X, training_Y, validation_X, validation_Y


def cat_train_validate_on_holdout(
    logger,
    run_id,
    training,
    validation,
    predictors,
    target,
    cat_features,
    params,
    test_X=None,
    label_name="",
    log_target=False,
):
    """Train a CatBoost model, validate on holdout data.

       If `test_X` has a valid value, creates a new model with number of best iteration
       found during holdout phase using training as well as validation data.

       Args:
            logger: Logger to be used
            training: Training DataFrame
            validation: Validation DataFrame
            predictors: List of names of features
            target: Name of target variable
            params: Parameters for CatBoost
            test_X: Test DataFrame

       Returns:
            bst: CatBoost model
            valid_score: Best validation score
            best_iteration: Value of best iteration
            prediction: Prediction generated on `test_X`
    """
    result_dict = {}
    logger.info("Training using CatBoost and validating on holdout")
    train_X, train_Y, validation_X, validation_Y = _get_x_y_from_training_validation(
        logger, training, validation, predictors, target
    )

    logger.info(
        (
            f"Shape of train_X, train_Y, validation_X, validation_Y: "
            f"{train_X.shape}, {train_Y.shape}, {validation_X.shape}, {validation_Y.shape}"
        )
    )

    if log_target:
        train_pool = Pool(
            data=train_X,
            label=np.log1p(train_Y),
            feature_names=predictors,
            cat_features=cat_features,
        )
        valid_pool = Pool(
            data=validation_X,
            label=np.log1p(validation_Y),
            feature_names=predictors,
            cat_features=cat_features,
        )
    else:
        train_pool = Pool(
            data=train_X,
            label=train_Y,
            feature_names=predictors,
            cat_features=cat_features,
        )
        valid_pool = Pool(
            data=validation_X,
            label=validation_Y,
            feature_names=predictors,
            cat_features=cat_features,
        )

    model = CatBoost(params=params)
    # List of categorical features have already been passed as a part of Pool
    # above. No need to pass via the argument of fit()
    model.fit(X=train_pool, eval_set=[train_pool, valid_pool], use_best_model=True)

    best_iteration = model.get_best_iteration()

    if log_target:
        valid_prediction = np.expm1(model.predict(valid_pool))
    else:
        valid_prediction = model.predict(valid_pool)

    valid_score = _calculate_perf_metric(validation_Y, valid_prediction)
    logger.info(f"Validation Score {valid_score}")
    logger.info(f"Best Iteration {best_iteration}")

    del train_pool, valid_pool, train_X, train_Y, validation_X, validation_Y
    gc.collect()

    if test_X is not None:
        logger.info("Retraining on the entire data including validation")
        training = pd.concat([training, validation])
        train_X, train_Y = _get_x_y_from_data(logger, training, predictors, target)
        logger.info(
            (f"Shape of train_X, train_Y: " f"{train_X.shape}, {train_Y.shape}")
        )

        if log_target:
            train_pool = Pool(
                data=train_X,
                label=np.log1p(train_Y),
                feature_names=predictors,
                cat_features=cat_features,
            )
        else:
            train_pool = Pool(
                data=train_X,
                label=train_Y,
                feature_names=predictors,
                cat_features=cat_features,
            )
        test_pool = Pool(
            data=test_X, feature_names=predictors, cat_features=cat_features
        )

        # Why?
        params.pop("eval_metric")
        params.pop("early_stopping_rounds")
        params.pop("use_best_model")
        params["n_estimators"] = best_iteration

        logger.info(f"Modified parameters for final model training.. {params}")

        model = CatBoost(params=params)
        model.fit(X=train_pool)

        logger.info(f"Predicting on test data: {test_X.shape}")
        if log_target:
            prediction = np.expm1(model.predict(test_pool))
        else:
            prediction = model.predict(test_pool)

        result_dict = _evaluate_and_log_for_holdout(
            logger=logger,
            run_id=run_id,
            valid_prediction=valid_prediction,
            valid_score=valid_score,
            y_predicted=prediction,
            result_dict=result_dict,
            best_iteration=best_iteration,
            label_name=label_name,
        )

        feature_importance = model.get_feature_importance()
        result_dict = _capture_feature_importance_for_holdout(
            feature_importance=feature_importance,
            features=predictors,
            result_dict=result_dict,
        )
    logger.info("Training/Prediction completed!")
    return result_dict


def xgb_train_validate_on_holdout(
    logger,
    run_id,
    training,
    validation,
    predictors,
    target,
    params,
    test_X=None,
    n_estimators=10000,
    early_stopping_rounds=100,
    verbose_eval=100,
    label_name="",
    log_target=False,
):
    """Train a XGBoost model, validate on holdout data. If `test_X`
       has a valid value, creates a new model with number of best iteration
       found during holdout phase using training as well as validation data.

       Args:
            logger: Logger to be used
            training: Training DataFrame
            validation: Validation DataFrame
            predictors: List of names of features
            target: Name of target variable
            params: Parameters for XGBoost
            test_X: Test DataFrame

       Returns:
            bst: XGB Booster object
            valid_score: Best validation score
            best_iteration: Value of best iteration
            prediction: Prediction generated on `test_X`
    """
    result_dict = {}
    logger.info("Training using XGBoost and validating on holdout")
    train_X, train_Y, validation_X, validation_Y = _get_x_y_from_training_validation(
        logger, training, validation, predictors, target
    )

    logger.info(
        (
            f"Shape of train_X, train_Y, validation_X, validation_Y: "
            f"{train_X.shape}, {train_Y.shape}, {validation_X.shape}, {validation_Y.shape}"
        )
    )

    if log_target:
        dtrain = xgb.DMatrix(
            data=train_X, label=np.log1p(train_Y), feature_names=predictors
        )
        dvalid = xgb.DMatrix(
            data=validation_X, label=np.log1p(validation_Y), feature_names=predictors
        )
    else:
        dtrain = xgb.DMatrix(data=train_X, label=train_Y, feature_names=predictors)
        dvalid = xgb.DMatrix(
            data=validation_X, label=validation_Y, feature_names=predictors
        )

    watchlist = [(dtrain, "train"), (dvalid, "valid_data")]
    bst = xgb.train(
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        params=params,
        verbose_eval=verbose_eval,
    )

    if log_target:
        valid_prediction = np.expm1(
            bst.predict(
                xgb.DMatrix(validation_X, feature_names=predictors),
                ntree_limit=bst.best_ntree_limit,
            )
        )
    else:
        valid_prediction = bst.predict(
            xgb.DMatrix(validation_X, feature_names=predictors),
            ntree_limit=bst.best_ntree_limit,
        )

    # Get best iteration
    best_iteration = bst.best_ntree_limit

    valid_score = _calculate_perf_metric(validation_Y, valid_prediction)
    logger.info(f"Validation Score {valid_score}")
    logger.info(f"Best Iteration {best_iteration}")

    del watchlist, dtrain, dvalid, train_X, train_Y, validation_X, validation_Y
    gc.collect()

    if test_X is not None:
        logger.info("Retraining on the entire data including validation")
        training = pd.concat([training, validation])
        train_X, train_Y = _get_x_y_from_data(logger, training, predictors, target)
        logger.info(
            (f"Shape of train_X, train_Y: " f"{train_X.shape}, {train_Y.shape}")
        )

        if log_target:
            dtrain = xgb.DMatrix(
                data=train_X, label=np.log1p(train_Y), feature_names=predictors
            )
            dtest = xgb.DMatrix(data=test_X, feature_names=predictors)
        else:
            dtrain = xgb.DMatrix(data=train_X, label=train_Y, feature_names=predictors)
            dtest = xgb.DMatrix(data=test_X, feature_names=predictors)

        bst = xgb.train(
            dtrain=dtrain,
            num_boost_round=best_iteration,
            params=params,
            verbose_eval=verbose_eval,
        )

        logger.info(f"Predicting on test data: {test_X.shape}")
        if log_target:
            prediction = np.expm1(bst.predict(dtest, ntree_limit=best_iteration))
        else:
            prediction = bst.predict(dtest, ntree_limit=best_iteration)

        result_dict = _evaluate_and_log_for_holdout(
            logger=logger,
            run_id=run_id,
            valid_prediction=valid_prediction,
            valid_score=valid_score,
            y_predicted=prediction,
            result_dict=result_dict,
            best_iteration=best_iteration,
            label_name=label_name,
        )

        # XGB may not use all the features while building the
        # model. Consider only the useful features
        feature_importance_values = bst.get_score().values()
        feature_importance_features = bst.get_score().keys()
        result_dict = _capture_feature_importance_for_holdout(
            feature_importance=feature_importance_values,
            features=feature_importance_features,
            result_dict=result_dict,
        )
    logger.info("Training/Prediction completed!")
    return result_dict


def lgb_train_validate_on_holdout(
    logger,
    run_id,
    training,
    validation,
    test_X,
    predictors,
    target,
    params,
    n_estimators=10000,
    early_stopping_rounds=100,
    cat_features="auto",
    verbose_eval=100,
    label_name="",
    log_target=False,
):
    """Train a LGB model and validate on holdout data.

       Args:
            logger: Logger to be used
            training: Training DataFrame
            validation: Validation DataFrame
            predictors: List of names of features
            target: Name of target variable
            params: Parameters for LGBM
            test_X: Test DataFrame

       Returns:
            bst: LGB Booster object
            valid_score: Best validation score
            best_iteration: Value of best iteration
            prediction: Prediction generated on `test_X`
    """
    result_dict = {}
    logger.info("Training using LGB and validating on holdout")
    train_X, train_Y, validation_X, validation_Y = _get_x_y_from_training_validation(
        logger, training, validation, predictors, target
    )

    logger.info(
        (
            f"Shape of train_X, train_Y, validation_X, validation_Y: "
            f"{train_X.shape}, {train_Y.shape}, {validation_X.shape}, {validation_Y.shape}"
        )
    )

    if log_target:
        dtrain = lgb.Dataset(train_X, label=np.log1p(train_Y))
        dvalid = lgb.Dataset(validation_X, np.log1p(validation_Y))
    else:
        dtrain = lgb.Dataset(train_X, label=train_Y)
        dvalid = lgb.Dataset(validation_X, validation_Y)

    bst = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        verbose_eval=verbose_eval,
        num_boost_round=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        feature_name=predictors,
        categorical_feature=cat_features,
    )

    best_iteration = bst.best_iteration

    if log_target:
        valid_prediction = np.expm1(
            bst.predict(validation_X, num_iteration=best_iteration)
        )
    else:
        valid_prediction = bst.predict(validation_X, num_iteration=best_iteration)

    valid_score = _calculate_perf_metric(validation_Y, valid_prediction)
    logger.info(f"Validation Score {valid_score}")
    logger.info(f"Best Iteration {best_iteration}")

    del dtrain, dvalid, train_X, train_Y, validation_X, validation_Y
    gc.collect()

    if test_X is not None:
        logger.info("Retraining on the entire data including validation")
        training = pd.concat([training, validation])
        train_X, train_Y = _get_x_y_from_data(logger, training, predictors, target)
        logger.info(
            (f"Shape of train_X, train_Y: " f"{train_X.shape}, {train_Y.shape}")
        )

        if log_target:
            dtrain = lgb.Dataset(train_X, label=np.log1p(train_Y))
        else:
            dtrain = lgb.Dataset(train_X, label=train_Y)

        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=best_iteration,
            verbose_eval=verbose_eval,
            feature_name=predictors,
            categorical_feature=cat_features,
        )

        logger.info(f"Predicting on test data: {test_X.shape}")
        if log_target:
            prediction = np.expm1(
                bst.predict(test_X[predictors], num_iteration=best_iteration)
            )
        else:
            prediction = bst.predict(test_X[predictors], num_iteration=best_iteration)

        result_dict = _evaluate_and_log_for_holdout(
            logger=logger,
            run_id=run_id,
            valid_prediction=valid_prediction,
            valid_score=valid_score,
            y_predicted=prediction,
            result_dict=result_dict,
            best_iteration=best_iteration,
            label_name=label_name,
        )

        feature_importance = bst.feature_importance()
        result_dict = _capture_feature_importance_for_holdout(
            feature_importance=feature_importance,
            features=predictors,
            result_dict=result_dict,
        )
    logger.info("Training/Prediction completed!")
    return result_dict


def _get_X_Y_from_CV(train_X, train_Y, train_index, validation_index):
    X_train, X_validation = (
        train_X.iloc[train_index].values,
        train_X.iloc[validation_index].values,
    )
    y_train, y_validation = (
        train_Y.iloc[train_index].values,
        train_Y.iloc[validation_index].values,
    )
    return X_train, X_validation, y_train, y_validation


def _get_X_Y_DF_from_CV(train_X, train_Y, train_index, validation_index):
    X_train, X_validation = (
        train_X.iloc[train_index],
        train_X.iloc[validation_index],
    )
    y_train, y_validation = (
        train_Y.iloc[train_index],
        train_Y.iloc[validation_index],
    )
    return X_train, X_validation, y_train, y_validation


def _capture_feature_importance_for_holdout(feature_importance, features, result_dict):
    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = features
    feature_importance_df["importance"] = feature_importance
    feature_importance_df = feature_importance_df.sort_values(
        by=["importance"], ascending=False
    )
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    result_dict["feature_importance"] = feature_importance_df
    return result_dict


def _capture_feature_importance_on_fold(
    feature_importance_df, features, feature_importance_on_fold, fold
):
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = feature_importance_on_fold
    fold_importance["fold"] = fold
    fold_importance.sort_values(by=["importance"], ascending=False, inplace=True)
    fold_importance.reset_index(drop=True, inplace=True)
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)
    return feature_importance_df


def _capture_feature_importance(
    feature_importance_df, n_important_features, result_dict
):
    """
    Identifies top `n_important_features` from `feature_importance_df` and then
    add those to `result_dict`
    """
    top_imp_features_df = (
        feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        # .sort_values(by="importance", ascending=False)[:n_important_features]
        .sort_values(by="importance", ascending=False)
        .reset_index()
    )
    result_dict["feature_importance"] = feature_importance_df
    result_dict["best_features"] = top_imp_features_df
    return result_dict


def _evaluate_and_log_for_holdout(
    logger,
    run_id,
    valid_prediction,
    valid_score,
    y_predicted,
    result_dict,
    best_iteration,
    label_name="",
):
    valid_score = round(valid_score, 5)

    result_dict["y_validation"] = valid_prediction
    result_dict["valid_score"] = valid_score
    result_dict["prediction"] = y_predicted

    util.update_tracking(
        run_id,
        f"valid_score_{label_name}",
        valid_score,
        is_integer=False,
        no_of_digits=5,
    )

    # Best Iteration
    if best_iteration:
        util.update_tracking(
            run_id,
            f"best_iteration_{label_name}",
            best_iteration,
            is_integer=False,
            no_of_digits=5,
        )

    return result_dict


def _evaluate_and_log(
    logger,
    run_id,
    train_Y,
    y_oof,
    y_predicted,
    metric,
    n_folds,
    result_dict,
    cv_scores,
    best_iterations,
    label_name="",
):
    y_predicted /= n_folds

    oof_score = round(_calculate_perf_metric(metric, train_Y, y_oof), 5)
    avg_cv_scores = round(sum(cv_scores) / len(cv_scores), 5)
    std_cv_scores = round(np.array(cv_scores).std(), 5)

    logger.info(f"Combined OOF score : {oof_score}")
    logger.info(f"Average of {n_folds} folds OOF score {avg_cv_scores}")
    logger.info(f"std of {n_folds} folds OOF score {std_cv_scores}")

    result_dict["y_oof"] = y_oof
    result_dict["prediction"] = y_predicted
    result_dict["oof_score"] = oof_score
    result_dict["cv_scores"] = cv_scores
    result_dict["avg_cv_scores"] = avg_cv_scores
    result_dict["std_cv_scores"] = std_cv_scores

    util.update_tracking(
        run_id, f"oof_score_{label_name}", oof_score, is_integer=False, no_of_digits=5
    )
    util.update_tracking(
        run_id,
        f"cv_avg_score_{label_name}",
        avg_cv_scores,
        is_integer=False,
        no_of_digits=5,
    )
    util.update_tracking(
        run_id,
        f"cv_std_score_{label_name}",
        std_cv_scores,
        is_integer=False,
        no_of_digits=5,
    )
    # Best Iteration
    if best_iterations:
        util.update_tracking(
            run_id,
            f"avg_best_iteration_{label_name}",
            np.mean(best_iterations),
            is_integer=False,
            no_of_digits=5,
        )
        util.update_tracking(
            run_id,
            f"std_best_iteration_{label_name}",
            np.std(best_iterations),
            is_integer=False,
            no_of_digits=5,
        )

    return result_dict


def xgb_train_validate_on_cv(
    logger,
    run_id,
    train_X,
    train_Y,
    test_X,
    metric,
    kf,
    features,
    params={},
    n_estimators=1000,
    early_stopping_rounds=100,
    verbose_eval=100,
    num_class=None,
    log_target=False,
    feval=None,
):
    """Train a XGBoost model, validate using cross validation. If `test_X` has
    a valid value, creates a new model with number of best iteration found during
    holdout phase using training as well as validation data.
    """
    if num_class:
        # This should be true for multiclass classification problems
        y_oof = np.zeros(shape=(len(train_X), num_class))
        y_predicted = np.zeros(shape=(len(test_X), num_class))
    else:
        y_oof = np.zeros(shape=(len(train_X)))
        y_predicted = np.zeros(shape=(len(test_X)))

    cv_scores = []
    result_dict = {}
    feature_importance = pd.DataFrame()
    best_iterations = []

    fold = 0
    n_folds = kf.get_n_splits()
    for train_index, validation_index in kf.split(train_X[features], y=train_Y):
        fold += 1
        logger.info(f"fold {fold} of {n_folds}")

        X_train, X_validation, y_train, y_validation = _get_X_Y_from_CV(
            train_X, train_Y, train_index, validation_index
        )

        if log_target:
            xgb_train = xgb.DMatrix(
                data=X_train, label=np.log1p(y_train), feature_names=features
            )
            xgb_eval = xgb.DMatrix(
                data=X_validation, label=np.log1p(y_validation), feature_names=features
            )
        else:
            xgb_train = xgb.DMatrix(data=X_train, label=y_train, feature_names=features)
            xgb_eval = xgb.DMatrix(
                data=X_validation, label=y_validation, feature_names=features
            )

        watchlist = [(xgb_train, "train"), (xgb_eval, "valid_data")]
        if feval:
            # Use the custom metrics
            # Comment out eval_metric in the parameters
            model = xgb.train(
                dtrain=xgb_train,
                num_boost_round=n_estimators,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                params=params,
                verbose_eval=verbose_eval,
                feval=feval,
            )
        else:
            model = xgb.train(
                dtrain=xgb_train,
                num_boost_round=n_estimators,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                params=params,
                verbose_eval=verbose_eval,
            )

        del xgb_train, xgb_eval, train_index, X_train, y_train
        gc.collect()

        if log_target:
            y_oof[validation_index] = np.expm1(
                model.predict(
                    xgb.DMatrix(X_validation, feature_names=features),
                    ntree_limit=model.best_ntree_limit,
                )
            )
        else:
            y_oof[validation_index] = model.predict(
                xgb.DMatrix(X_validation, feature_names=features),
                ntree_limit=model.best_ntree_limit,
            )
        if test_X is not None:
            xgb_test = xgb.DMatrix(test_X.values, feature_names=features)
            if log_target:
                y_predicted += np.expm1(
                    model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
                )
            else:
                y_predicted += model.predict(
                    xgb_test, ntree_limit=model.best_ntree_limit
                )

        best_iteration = model.best_ntree_limit
        best_iterations.append(best_iteration)
        logger.info(f"Best number of iterations for fold {fold} is: {best_iteration}")

        cv_oof_score = _calculate_perf_metric(
            metric, y_validation, y_oof[validation_index]
        )
        cv_scores.append(cv_oof_score)
        logger.info(f"CV OOF Score for fold {fold} is {cv_oof_score}")

        del validation_index, X_validation, y_validation
        gc.collect()

        # XGB may not use all the features while building the
        # model. Consider only the useful features
        feature_importance_on_fold_values = model.get_score().values()
        feature_importance_on_fold_keys = model.get_score().keys()

        feature_importance = _capture_feature_importance_on_fold(
            feature_importance,
            feature_importance_on_fold_keys,
            feature_importance_on_fold_values,
            fold,
        )

        # util.update_tracking(
        #     run_id, "metric_fold_{}".format(fold), cv_oof_score, is_integer=False
        # )

    result_dict = _evaluate_and_log(
        logger,
        run_id,
        train_Y,
        y_oof,
        y_predicted,
        metric,
        n_folds,
        result_dict,
        cv_scores,
        best_iterations,
    )

    del y_oof
    gc.collect()

    result_dict = _capture_feature_importance(
        feature_importance, n_important_features=10, result_dict=result_dict
    )

    logger.info("Training/Prediction completed!")
    return result_dict


def lgb_train_validate_on_cv(
    logger,
    run_id,
    train_X,
    train_Y,
    test_X,
    metric,
    kf,
    features,
    params={},
    n_estimators=1000,
    early_stopping_rounds=100,
    cat_features="auto",
    verbose_eval=100,
    num_class=None,
    log_target=False,
    retrain=False,
    feval=None,
):
    """Train a LightGBM model, validate using cross validation. If `test_X` has
    a valid value, creates a new model with number of best iteration found during
    holdout phase using training as well as validation data.

    startify_by_labels: Used as the label for StartifiedKFold on top of continous
    variables
    """
    cv_scores = []
    result_dict = {}
    feature_importance = pd.DataFrame()
    best_iterations = []

    if num_class:
        # This should be true for multiclass classification problems
        y_oof = np.zeros(shape=(len(train_X), num_class))
        y_predicted = np.zeros(shape=(len(test_X), num_class))
    else:
        y_oof = np.zeros(shape=(len(train_X)))
        y_predicted = np.zeros(shape=(len(test_X)))

    fold = 0
    n_folds = kf.get_n_splits()
    for train_index, validation_index in kf.split(X=train_X, y=train_Y):
        fold += 1
        logger.info(f"fold {fold} of {n_folds}")

        if num_class:
            logger.info(f"Number of classes in target {train_Y.nunique()}")

        X_train, X_validation, y_train, y_validation = _get_X_Y_DF_from_CV(
            train_X, train_Y, train_index, validation_index
        )

        if log_target:
            lgb_train = lgb.Dataset(X_train, np.log1p(y_train))
            lgb_eval = lgb.Dataset(
                X_validation, np.log1p(y_validation), reference=lgb_train
            )
        else:
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

        if feval:
            # For custom metric. metric should be set to "custom" in parameters
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=verbose_eval,
                early_stopping_rounds=early_stopping_rounds,
                num_boost_round=n_estimators,
                feature_name=features,
                categorical_feature=cat_features,
                feval=feval,
            )
        else:
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=verbose_eval,
                early_stopping_rounds=early_stopping_rounds,
                num_boost_round=n_estimators,
                feature_name=features,
                categorical_feature=cat_features,
            )

        del lgb_train, lgb_eval, train_index, X_train, y_train
        gc.collect()

        if log_target:
            y_oof[validation_index] = np.expm1(
                model.predict(X_validation, num_iteration=model.best_iteration)
            )
        else:
            y_oof[validation_index] = model.predict(
                X_validation, num_iteration=model.best_iteration
            )

        best_iteration = model.best_iteration
        best_iterations.append(best_iteration)
        logger.info(f"Best number of iterations for fold {fold} is: {best_iteration}")

        cv_oof_score = _calculate_perf_metric(
            metric, y_validation, y_oof[validation_index]
        )
        cv_scores.append(cv_oof_score)
        logger.info(f"CV OOF Score for fold {fold} is {cv_oof_score}")

        del validation_index, X_validation, y_validation
        gc.collect()

        feature_importance_on_fold = model.feature_importance()
        feature_importance = _capture_feature_importance_on_fold(
            feature_importance, features, feature_importance_on_fold, fold
        )

        # util.update_tracking(
        #     run_id, f"metric_fold_{fold}", cv_oof_score, is_integer=False
        # )
        if retrain:
            params["seed"] = _get_random_seeds(fold)
            logger.info(
                f"Retraining the model with seed [{params['seed']}] on full data"
            )
            if log_target:
                lgb_train = lgb.Dataset(train_X, np.log1p(train_Y))
            else:
                lgb_train = lgb.Dataset(train_X, train_Y)

            if feval:
                # For custom metric. metric should be set to "custom" in parameters
                model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=model.best_iteration,
                    feature_name=features,
                    categorical_feature=cat_features,
                    feval=feval,
                )
            else:
                model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=model.best_iteration,
                    feature_name=features,
                    categorical_feature=cat_features,
                )

        if test_X is not None:
            if log_target:
                y_predicted += np.expm1(
                    model.predict(test_X.values, num_iteration=model.best_iteration)
                )
            else:
                y_predicted += model.predict(
                    test_X.values, num_iteration=model.best_iteration
                )
                logger.info(f"y_predicted {y_predicted}")

    result_dict = _evaluate_and_log(
        logger,
        run_id,
        train_Y,
        y_oof,
        y_predicted,
        metric,
        n_folds,
        result_dict,
        cv_scores,
        best_iterations,
    )

    del y_oof
    gc.collect()

    result_dict = _capture_feature_importance(
        feature_importance, n_important_features=10, result_dict=result_dict
    )

    logger.info("Training/Prediction completed!")
    return result_dict


def lgb_train_validate_on_cv_mean_encoding(
    logger,
    run_id,
    train_X,
    train_Y,
    test_X,
    metric,
    kf,
    features,
    params={},
    n_estimators=1000,
    early_stopping_rounds=100,
    cat_features="auto",
    verbose_eval=100,
    num_class=None,
    log_target=False,
    retrain=False,
    feval=None,
    target_val=None,
    cat_enc_cols=None,
):
    """Train a LightGBM model, validate using cross validation. If `test_X` has
    a valid value, creates a new model with number of best iteration found during
    holdout phase using training as well as validation data.

    startify_by_labels: Used as the label for StartifiedKFold on top of continous
    variables
    """
    cv_scores = []
    result_dict = {}
    feature_importance = pd.DataFrame()
    best_iterations = []

    if num_class:
        # This should be true for multiclass classification problems
        y_oof = np.zeros(shape=(len(train_X), num_class))
        y_predicted = np.zeros(shape=(len(test_X), num_class))
    else:
        y_oof = np.zeros(shape=(len(train_X)))
        y_predicted = np.zeros(shape=(len(test_X)))

    for col in cat_enc_cols:
        features = features + [f"{col}_m_enc"]

    fold = 0
    n_folds = kf.get_n_splits()
    for train_index, validation_index in kf.split(X=train_X, y=train_Y):
        fold += 1
        logger.info(f"fold {fold} of {n_folds}")

        if num_class:
            logger.info(f"Number of classes in target {train_Y.nunique()}")

        X_train, X_validation, y_train, y_validation = _get_X_Y_DF_from_CV(
            train_X, train_Y, train_index, validation_index
        )

        for col in cat_enc_cols:
            # create dict of category:mean target
            X_temp = pd.concat([X_train, y_train], axis=1)
            mapping_dict = dict(X_temp.groupby(col)[target_val].mean())
            X_train[f"{col}_m_enc"] = X_train[col].map(mapping_dict)
            X_validation[f"{col}_m_enc"] = X_validation[col].map(mapping_dict)
            test_X[f"{col}_m_enc"] = test_X[col].map(mapping_dict)

        if log_target:
            lgb_train = lgb.Dataset(X_train, np.log1p(y_train))
            lgb_eval = lgb.Dataset(
                X_validation, np.log1p(y_validation), reference=lgb_train
            )
        else:
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

        if feval:
            # For custom metric. metric should be set to "custom" in parameters
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=verbose_eval,
                early_stopping_rounds=early_stopping_rounds,
                num_boost_round=n_estimators,
                feature_name=features,
                categorical_feature=cat_features,
                feval=feval,
            )
        else:
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=verbose_eval,
                early_stopping_rounds=early_stopping_rounds,
                num_boost_round=n_estimators,
                feature_name=features,
                categorical_feature=cat_features,
            )

        del lgb_train, lgb_eval, train_index, X_train, y_train
        gc.collect()

        if log_target:
            y_oof[validation_index] = np.expm1(
                model.predict(X_validation, num_iteration=model.best_iteration)
            )
        else:
            y_oof[validation_index] = model.predict(
                X_validation, num_iteration=model.best_iteration
            )

        best_iteration = model.best_iteration
        best_iterations.append(best_iteration)
        logger.info(f"Best number of iterations for fold {fold} is: {best_iteration}")

        cv_oof_score = _calculate_perf_metric(
            metric, y_validation, y_oof[validation_index]
        )
        cv_scores.append(cv_oof_score)
        logger.info(f"CV OOF Score for fold {fold} is {cv_oof_score}")

        del validation_index, X_validation, y_validation
        gc.collect()

        feature_importance_on_fold = model.feature_importance()
        feature_importance = _capture_feature_importance_on_fold(
            feature_importance, features, feature_importance_on_fold, fold
        )

        # util.update_tracking(
        #     run_id, f"metric_fold_{fold}", cv_oof_score, is_integer=False
        # )
        if retrain:
            params["seed"] = _get_random_seeds(fold)
            logger.info(
                f"Retraining the model with seed [{params['seed']}] on full data"
            )
            if log_target:
                lgb_train = lgb.Dataset(train_X, np.log1p(train_Y))
            else:
                lgb_train = lgb.Dataset(train_X, train_Y)

            if feval:
                # For custom metric. metric should be set to "custom" in parameters
                model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=model.best_iteration,
                    feature_name=features,
                    categorical_feature=cat_features,
                    feval=feval,
                )
            else:
                model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=model.best_iteration,
                    feature_name=features,
                    categorical_feature=cat_features,
                )

        if test_X is not None:
            if log_target:
                y_predicted += np.expm1(
                    model.predict(test_X.values, num_iteration=model.best_iteration)
                )
            else:
                y_predicted += model.predict(
                    test_X.values, num_iteration=model.best_iteration
                )

    result_dict = _evaluate_and_log(
        logger,
        run_id,
        train_Y,
        y_oof,
        y_predicted,
        metric,
        n_folds,
        result_dict,
        cv_scores,
        best_iterations,
    )

    del y_oof
    gc.collect()

    result_dict = _capture_feature_importance(
        feature_importance, n_important_features=10, result_dict=result_dict
    )

    logger.info("Training/Prediction completed!")
    return result_dict


def permutation_importance(model, X, y):
    """
    Custom function for calculating permutation importance

    https://www.kaggle.com/c/ieee-fraud-detection/discussion/107877

    The lower the score means that the feature is important.
    If the score is positive that feature might be hurting your model.
    """
    perm = {}
    y_predicted = model.predict(X, num_iteration=model.best_iteration)
    baseline = roc_auc_score(y, y_predicted)
    for col_name in X.columns:
        value = X[col_name].copy()
        X.loc[:, col_name] = np.random.permutation(X[col_name].values)
        y_predicted = model.predict(X, num_iteration=model.best_iteration)
        perm[col_name] = roc_auc_score(y, y_predicted) - baseline
        X.loc[:, col_name] = value
    return perm


def _capture_permutation_importance_on_fold(
    permutation_importance_df, permutation_importance_on_fold, fold
):
    # Add an attribute called fold
    permutation_importance_on_fold["fold"] = fold

    # Append it to parent DF
    perm_importance_df = pd.concat(
        [permutation_importance_df, permutation_importance_on_fold], axis=0
    )
    return perm_importance_df


def _capture_permutation_importance(perm_importance_df, n_important_features=None):
    """
    """
    top_imp_features_df = (
        perm_importance_df[["feature", "weight"]]
        .groupby("feature")
        .mean()
        # .sort_values(by="importance", ascending=False)[:n_important_features]
        .sort_values(by="weight", ascending=False)
        .reset_index()
    )
    return top_imp_features_df


def lgb_train_perm_importance_on_cv(
    logger,
    train_X,
    train_Y,
    metric,
    kf,
    features,
    seed,
    params={},
    early_stopping_rounds=100,
    cat_features="auto",
    verbose_eval=100,
    display_imp=False,
    feval=None,
):
    """Train a LightGBM model and computes permutation importance
    across multiple folds of CV

    Returns a DataFrame consisting of average value of permutation importance
    for different features computed across multiple folds of CV
    """
    permutation_importance_df = pd.DataFrame()

    fold = 0
    n_folds = kf.get_n_splits()
    for train_index, validation_index in kf.split(X=train_X, y=train_Y):
        fold += 1
        logger.info(f"Starting fold {fold} of {n_folds}")

        X_train, X_validation, y_train, y_validation = _get_X_Y_from_CV(
            train_X, train_Y, train_index, validation_index
        )

        model = lgb.LGBMClassifier(**params)

        if feval:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_validation, y_validation)],
                verbose=verbose_eval,
                early_stopping_rounds=early_stopping_rounds,
                feature_name=features,
                categorical_feature=cat_features,
                # For the train API, name of the parameter
                # is feval. For sklearn api, it is eval_metric
                eval_metric=feval,
            )
        else:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_validation, y_validation)],
                verbose=verbose_eval,
                early_stopping_rounds=early_stopping_rounds,
                feature_name=features,
                categorical_feature=cat_features,
            )

        del X_train, y_train, train_index, validation_index
        gc.collect()

        scorer = _get_scorer(metric)

        # calculate permitation importance for the classifier
        perm = eli5.sklearn.PermutationImportance(
            model,
            scoring=make_scorer(score_func=scorer, average="weighted"),
            random_state=seed,
        ).fit(X_validation, y_validation, num_iteration=model.best_iteration_)

        if display_imp:
            display(eli5.show_weights(perm, feature_names=features, top=None))

        expl = eli5.explain_weights(perm, feature_names=features, top=None)
        permutation_importance_df_fold = eli5.format_as_dataframe(expl)
        permutation_importance_df = _capture_permutation_importance_on_fold(
            permutation_importance_df, permutation_importance_df_fold, fold
        )

        del X_validation, y_validation
        gc.collect()

    top_imp_features_df = _capture_permutation_importance(
        permutation_importance_df, n_important_features=10
    )

    return permutation_importance_df, top_imp_features_df


def cat_train_validate_on_cv(
    logger,
    run_id,
    train_X,
    train_Y,
    test_X,
    metric,
    kf,
    features,
    params={},
    num_class=None,
    cat_features=None,
    log_target=False,
):
    """Train a CatBoost model, validate using cross validation. If `test_X` has
    a valid value, creates a new model with number of best iteration found during
    holdout phase using training as well as validation data.

    Note: For CatBoost, categorical features need to be in String or Category data type.
    """
    if num_class:
        # This should be true for multiclass classification problems
        y_oof = np.zeros(shape=(len(train_X), num_class))
        y_predicted = np.zeros(shape=(len(test_X), num_class))
    else:
        y_oof = np.zeros(shape=(len(train_X)))
        y_predicted = np.zeros(shape=(len(test_X)))

    cv_scores = []
    result_dict = {}
    feature_importance = pd.DataFrame()
    best_iterations = []

    fold = 0
    n_folds = kf.get_n_splits()
    for train_index, validation_index in kf.split(train_X[features], train_Y):
        fold += 1
        logger.info(f"fold {fold} of {n_folds}")

        X_train, X_validation, y_train, y_validation = _get_X_Y_from_CV(
            train_X, train_Y, train_index, validation_index
        )

        if log_target:
            # feature_names accepts only list
            cat_train = Pool(
                data=X_train,
                label=np.log1p(y_train),
                feature_names=features,
                cat_features=cat_features,
            )
            cat_eval = Pool(
                data=X_validation,
                label=np.log1p(y_validation),
                feature_names=features,
                cat_features=cat_features,
            )
        else:
            # feature_names accepts only list
            cat_train = Pool(
                data=X_train,
                label=y_train,
                feature_names=features,
                cat_features=cat_features,
            )
            cat_eval = Pool(
                data=X_validation,
                label=y_validation,
                feature_names=features,
                cat_features=cat_features,
            )

        model = CatBoost(params=params)
        # List of categorical features have already been passed as a part of Pool
        # above. No need to pass via the argument of fit()
        model.fit(cat_train, eval_set=cat_eval, use_best_model=True)

        del train_index, X_train, y_train, cat_train
        gc.collect()

        if log_target:
            y_oof[validation_index] = np.expm1(model.predict(cat_eval))
        else:
            y_oof[validation_index] = model.predict(cat_eval)

        if test_X is not None:
            cat_test = Pool(
                data=test_X, feature_names=features, cat_features=cat_features
            )
            if log_target:
                y_predicted += np.expm1(model.predict(cat_test))
            else:
                y_predicted += model.predict(cat_test)

        del cat_eval, cat_test

        best_iteration = model.best_iteration_
        best_iterations.append(best_iteration)
        logger.info(f"Best number of iterations for fold {fold} is: {best_iteration}")

        cv_oof_score = _calculate_perf_metric(
            metric, y_validation, y_oof[validation_index]
        )
        cv_scores.append(cv_oof_score)
        logger.info(f"CV OOF Score for fold {fold} is {cv_oof_score}")

        del validation_index, X_validation, y_validation
        gc.collect()

        feature_importance_on_fold = model.get_feature_importance()
        feature_importance = _capture_feature_importance_on_fold(
            feature_importance, features, feature_importance_on_fold, fold
        )

        # util.update_tracking(
        #     run_id,
        #     "metric_fold_{}".format(fold),
        #     cv_oof_score,
        #     is_integer=False,
        #     no_of_digits=5,
        # )

    result_dict = _evaluate_and_log(
        logger,
        run_id,
        train_Y,
        y_oof,
        y_predicted,
        metric,
        n_folds,
        result_dict,
        cv_scores,
        best_iterations,
    )

    del y_oof
    gc.collect()

    result_dict = _capture_feature_importance(
        feature_importance, n_important_features=10, result_dict=result_dict
    )

    logger.info("Training/Prediction completed!")
    return result_dict


def sklearn_train_validate_on_cv(
    logger, run_id, sklearn_model, train_X, train_Y, test_X, kf, features, metric
):
    """
    Features should be a list
    """

    y_oof = np.zeros(len(train_X))
    y_predicted = np.zeros(len(test_X))
    cv_scores = []
    result_dict = {}

    fold = 0
    n_folds = kf.get_n_splits()
    for train_index, validation_index in kf.split(X=train_X[features], y=train_Y):
        fold += 1
        logger.info(f"fold {fold} of {n_folds}")

        X_train, X_validation, y_train, y_validation = _get_X_Y_from_CV(
            train_X, train_Y, train_index, validation_index
        )

        sklearn_model.fit(X_train, y_train)

        y_oof[validation_index] = sklearn_model.predict_proba(X_validation)[:, -1]
        if test_X is not None:
            y_predicted += sklearn_model.predict_proba(test_X.values)[:, -1]

        del X_train, y_train

        cv_oof_score = _calculate_perf_metric(
            metric, y_validation, y_oof[validation_index]
        )
        cv_scores.append(cv_oof_score)
        logger.info(f"CV OOF Score for fold {fold} is {cv_oof_score}")

        del validation_index, X_validation, y_validation
        gc.collect()

        # util.update_tracking(
        #     run_id, "metric_fold_{}".format(fold), cv_oof_score, is_integer=False
        # )

    result_dict = _evaluate_and_log(
        logger,
        run_id,
        train_Y,
        y_oof,
        y_predicted,
        metric,
        n_folds,
        result_dict,
        cv_scores,
        best_iterations=None,
    )

    del y_oof
    gc.collect()

    logger.info("Training/Prediction completed!")
    return result_dict
