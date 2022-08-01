import gc
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd

import src.config.constants as constants
import src.viz as viz

__all__ = [
    "set_seed",
    "trigger_gc",
    "set_timezone",
    "get_logger",
    "update_tracking",
    "save_file",
    "save_artifacts",
    "save_optuna_artifacts",
    "save_permutation_imp_artifacts",
    "calculate_final_score",
    "create_submission_file",
    "save_artifacts_holdout",
]


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def trigger_gc(logger):
    """
    Trigger GC
    """
    logger.info(f"Number of object collected [{gc.collect()}]")


def set_timezone():
    """
    Sets the time zone to Kolkata.
    """
    os.environ["TZ"] = "Asia/Calcutta"
    time.tzset()


def get_logger(logger_name, model_number=None, run_id=None, path=None):
    """
    Returns a logger with Stream & File Handler.
    File Handler is created only if model_number, run_id, path
    are not None.

    https://realpython.com/python-logging/
    https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480
    """
    FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    s_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(FORMAT)
    s_handler.setFormatter(formatter)
    logger.addHandler(s_handler)

    if all([model_number, run_id, path]):
        f_handler = logging.FileHandler(f"{path}/{model_number}_{run_id}.log")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger


def update_tracking(
    run_id,
    key,
    value,
    csv_file=constants.TRACKING_FILE,
    is_integer=False,
    no_of_digits=None,
    drop_incomplete_rows=False,
):
    """
    Function to update the tracking CSV with information about the model

    https://github.com/RobMulla/kaggle-ieee-fraud-detection/blob/master/scripts/M001.py#L98

    """
    try:
        df = pd.read_csv(csv_file, index_col=[0])

        # If the file exists, drop rows (without final results)
        # for previous runs which has been stopped inbetween.
        if drop_incomplete_rows & ("oof_score" in df.columns):
            df = df.loc[~df["oof_score"].isna()]

    except FileNotFoundError:
        df = pd.DataFrame()
        # df["lb_score"] = 0

    if is_integer:
        value = round(value)
    elif no_of_digits is not None:
        value = round(value, no_of_digits)

    # Model number is index
    df.loc[run_id, key] = value
    df.to_csv(csv_file)


def save_file(logger, df, dir_name, file_name, index=True, compression=None):
    """
    common method to save submission, off files etc.
    """
    logger.info(f"Saving {dir_name}/{file_name}")
    if compression:
        df.to_csv(
            f"{dir_name}/{file_name}", index=index, sep=",", compression=compression
        )
    else:
        df.to_csv(f"{dir_name}/{file_name}", index=index, sep=",")


def save_artifacts_holdout(
    logger,
    is_test,
    is_plot_fi,
    result_dict,
    model_number,
    run_id,
    oof_dir,
    fi_dir,
    fi_fig_dir,
    label_name="",
):
    """
    Save the submission, OOF predictions, feature importance values
    and plos to different directories.
    """
    score = result_dict["valid_score"]

    if is_test is False:
        # Save OOF
        oof_df = pd.DataFrame(result_dict["y_validation"])
        save_file(
            logger,
            oof_df,
            oof_dir,
            f"y_val_{label_name}_{model_number}_{run_id}_{score:.5f}.csv",
        )

    if is_plot_fi is True:
        # Feature Importance
        feature_importance_df = result_dict["feature_importance"]
        save_file(
            logger,
            feature_importance_df,
            fi_dir,
            f"fi_{label_name}_{model_number}_{run_id}_{score:.5f}.csv",
        )

        # Save the plot for best features
        best_features = result_dict["feature_importance"]
        viz.save_feature_importance_as_fig(
            best_features,
            fi_fig_dir,
            f"fi_{label_name}_{model_number}_{run_id}_{score:.5f}.png",
        )


def save_artifacts(
    logger,
    target,
    is_plot_fi,
    result_dict,
    submission_df,
    train_index,
    model_number,
    run_id,
    sub_dir,
    oof_dir,
    fi_dir,
    fi_fig_dir,
):
    """
    Save the submission, OOF predictions, feature importance values
    and plos to different directories.
    """
    score = result_dict["avg_cv_scores"]

    # Save submission file
    print(result_dict["prediction"])
    submission_df[target] = result_dict["prediction"]
    save_file(
        logger,
        submission_df,
        sub_dir,
        f"sub_{model_number}_{run_id}_{score:.5f}.gz",
        compression="gzip",
    )

    # Save OOF
    if train_index is not None:
        oof_df = pd.DataFrame(result_dict["y_oof"], index=train_index)
        save_file(
            logger, oof_df, oof_dir, f"oof_{model_number}_{run_id}_{score:.5f}.csv",
        )

    if is_plot_fi is True:
        # Feature Importance
        feature_importance_df = result_dict["feature_importance"]
        save_file(
            logger,
            feature_importance_df,
            fi_dir,
            f"fi_{model_number}_{run_id}_{score:.5f}.csv",
        )

        best_features_df = result_dict["best_features"]
        save_file(
            logger,
            best_features_df,
            fi_dir,
            f"bf_{model_number}_{run_id}_{score:.5f}.csv",
        )

        # Save the plot for best features
        best_features = result_dict["best_features"]
        viz.save_feature_importance_as_fig(
            best_features, fi_fig_dir, f"fi_{model_number}_{run_id}_{score:.5f}.png",
        )


def calculate_final_score(run_id, results_dict_co, results_dict_ben, results_dict_no):
    """
    Competition specific
    """
    agg_val_score = (
        results_dict_co["valid_score"]
        + results_dict_ben["valid_score"]
        + results_dict_no["valid_score"]
    ) / 3

    if agg_val_score:
        update_tracking(
            run_id, "agg_val_score", agg_val_score, is_integer=False, no_of_digits=5
        )

    return agg_val_score


def create_submission_file(
    logger,
    run_id,
    model_number,
    sub_dir,
    score,
    sub_df,
    test_df,
    results_dict_co,
    results_dict_ben,
    results_dict_no,
):
    """
    Competition specific
    """
    sub_df.date_time = test_df.date_time
    sub_df.target_carbon_monoxide = results_dict_co["prediction"]
    sub_df.target_benzene = results_dict_ben["prediction"]
    sub_df.target_nitrogen_oxides = results_dict_no["prediction"]

    save_file(
        logger,
        sub_df,
        sub_dir,
        f"sub_{model_number}_{run_id}_{score:.5f}.csv",
        index=False,
    )


def save_optuna_artifacts(
    logger, best_score, best_params, param_importance, run_id, hpo_dir
):
    """Saves the best params & best score returned by optuna in a CSV & png file
    """
    # Convert best params into a DF
    best_params_df = pd.DataFrame.from_dict(best_params, orient="index")
    best_params_df = best_params_df.reset_index(drop=False)
    best_params_df.columns = ["param_name", "param_value"]

    # Add best_score as a parameter
    best_params_df.loc[len(best_params_df.index)] = ["best_score", best_score]

    importance_df = pd.DataFrame.from_dict(
        param_importance, orient="index"
    ).reset_index()
    importance_df.columns = ["param_name", "importance"]

    params_df = pd.merge(best_params_df, importance_df, how="outer", on="param_name")

    # Save the file as a CSV
    file_name = f"hpo_optuna_{run_id}_{best_score:.5f}"
    save_file(
        logger=logger, df=params_df, dir_name=hpo_dir, file_name=f"{file_name}.csv"
    )

    # Save the parameter importance as an image.
    # First drop the best score. We don't need it for plotting
    params_df = params_df[params_df.param_name != "best_score"]
    viz.save_optuna_param_importance_as_fig(
        params_df=params_df, dir_name=hpo_dir, file_name=f"{file_name}.png"
    )


def save_permutation_imp_artifacts(
    logger, perm_imp_df, top_imp_df, run_id, model_name, fi_dir, fi_fig_dir
):
    """Saves the fi
    """
    # Save the file as a CSV
    all_fold_file_name = f"pi_all_fold_{model_name}_{run_id}"
    save_file(
        logger=logger,
        df=perm_imp_df,
        dir_name=fi_dir,
        file_name=f"{all_fold_file_name}.csv",
        index=False,
    )

    summary_file_name = f"pi_summarized_{model_name}_{run_id}"
    save_file(
        logger=logger,
        df=top_imp_df,
        dir_name=fi_dir,
        file_name=f"{summary_file_name}.csv",
        index=False,
    )

    # Save the plot for best features
    viz.save_permutation_importance_as_fig(
        top_imp_df, fi_fig_dir, f"{summary_file_name}.png"
    )
