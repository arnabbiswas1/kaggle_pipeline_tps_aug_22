import os
import gc
from datetime import datetime
import logging

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import optuna as optuna
import lightgbm as lgb

import common.com_util as common
import modeling.train_util as train_util
import munging.process_data_util as process_data
import config.constants as constants


def get_data(frac=None):
    TARGET = "loss"
    # Read the processed data. Read the features with which best result has been
    # received so far.
    train_df, test_df, sample_submission_df = process_data.read_processed_data(
        logger, constants.PROCESSED_DATA_DIR, train=True, test=True, sample_submission=True
    )

    if frac:
        train_df = train_df.sample(frac=frac)
        test_df = test_df.sample(frac=frac)

    cat_features = "f1", "f86", "f55"

    features_df = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/generated_features.parquet")
    logger.info(f"Shape of the features {features_df.shape}")

    combined_df = pd.concat([train_df.drop("loss", axis=1), test_df])
    orginal_features = list(test_df.columns)
    combined_df = pd.concat([combined_df, features_df], axis=1)

    logger.info(f"Shape of combined data with features {combined_df.shape}")
    feature_names = process_data.get_freq_encoding_feature_names(combined_df)

    logger.info(f"Selceting frequency encoding features {feature_names}")
    combined_df = combined_df.loc[:, orginal_features + feature_names]
    logger.info(f"Shape of the data after selecting features {combined_df.shape}")

    train_X = combined_df.iloc[0: len(train_df)]
    train_Y = train_df[TARGET]

    logger.debug(f"Shape of train_X: {train_X.shape}, train_Y: {train_Y.shape}")

    predictors = list(train_X.columns)

    return train_X, train_Y, predictors, cat_features


def objective(trial):
    # Param Definition from : https://www.kaggle.com/isaienkov/lgbm-optuna-rfe
    params = {
        "objective": "root_mean_squared_error",
        "metric": "RMSE",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,

        "n_estimators": trial.suggest_int("n_estimators", 20, 1000),
        "max_depth": trial.suggest_int('max_depth', 6, 10),
        "learning_rate": trial.suggest_uniform('learning_rate', 0.0001, 0.99),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 0.5, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 0.5, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.0001, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction",  0.0001, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 1200),
    }

    train_X, train_Y, predictors, cat_features = get_data()

    skf = StratifiedKFold(n_splits=10, shuffle=False)
    n_folds = skf.get_n_splits()
    fold = 0
    y_oof = np.zeros(len(train_X))
    cv_scores = []
    for train_index, validation_index in skf.split(X=train_X, y=train_Y):
        fold += 1
        logger.info(f"fold {fold} of {n_folds}")

        X_train, X_validation, y_train, y_validation = train_util.__get_X_Y_from_CV(
            train_X, train_Y, train_index, validation_index
        )

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(
            X_validation, y_validation, reference=lgb_train)

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_eval],
            verbose_eval=False,
            early_stopping_rounds=100,
            feature_name=predictors,
            categorical_feature=cat_features,
            callbacks=[pruning_callback]
        )

        del lgb_train, lgb_eval, train_index, X_train, y_train
        gc.collect()

        y_oof[validation_index] = model.predict(X_validation)

        cv_oof_score = train_util.__calculate_perf_metric(y_validation, y_oof[validation_index])
        logger.info(f"CV Score for fold {fold}: {cv_oof_score}")
        cv_scores.append(cv_oof_score)

    mean_cv_score = sum(cv_scores) / len(cv_scores)
    logger.info(f"Mean CV Score {mean_cv_score}")
    return mean_cv_score


if __name__ == "__main__":

    RUN_ID = datetime.now().strftime("%m%d_%H%M")
    MODEL_NAME = os.path.basename(__file__).split(".")[0]

    LOGGER_NAME = "hpo"
    logger = common.get_logger(LOGGER_NAME, MODEL_NAME, RUN_ID, constants.LOG_DIR)
    logger.setLevel(logging.WARNING)

    EARLY_STOPPING_ROUNDS = 100

    # Optimization
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=100), direction="maximize"
    )
    study.optimize(
        objective,
        # n_trials=5,
        timeout=3600*12
        )

    logger.warning(f"Number of finished trials: {len(study.trials)}")

    best_score = study.best_value
    best_params = study.best_params
    # A dictionary
    param_importance = optuna.importance.get_param_importances(study)

    logger.warning(f"Best score: {best_score}")
    logger.warning(f"Best params: {best_params}")
    logger.warning(f"Parameter importnace: {param_importance}")

    # Save the best params
    common.save_optuna_artifacts(
        logger,
        best_score=best_score,
        best_params=best_params,
        param_importance=param_importance,
        run_id=RUN_ID,
        hpo_dir=constants.HPO_DIR,
    )