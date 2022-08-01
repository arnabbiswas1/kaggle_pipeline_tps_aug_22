"""All the common methods related to cross validation
"""
import numpy as np
import pandas as pd

__all__ = [
    "get_data_splits_by_fraction",
    "get_data_splits_by_month",
    "get_data_splits_by_date_block",
    "create_stratify_labels_for_regression"
]


def get_data_splits_by_fraction(logger, df, valid_fraction=0.1):
    """
    Creating holdout set from the train data based on fraction
    """
    logger.info(
        f"Splitting the data into train and holdout with validation fraction {valid_fraction}..."
    )
    valid_size = int(len(df) * valid_fraction)
    train = df[:valid_size]
    validation = df[valid_size:]
    logger.info(f"Shape of the training data {train.shape} ")
    logger.info(f"Shape of the validation data {validation.shape}")
    return train, validation


def get_data_splits_by_month(logger, df, train_months, validation_months):
    """
    Returns training & holdout set from the train data based on months

    train_months: List of numbers representing months. For example,
                  [1, 2, 3, 4] represents Jan, Feb, March, April

    test_months: List of numbers representing months. For example,
                  [5, 6, 7, 8] represents May, June, July, August
    """
    logger.info("Splitting the data into train and holdout based on months...")
    logger.info(f"Training months {train_months}")
    logger.info(f"Validation months {validation_months}")
    training = df[df.month.isin(train_months)]
    validation = df[df.month.isin(validation_months)]
    logger.info(f"Shape of the training data {training.shape} ")
    logger.info(f"Shape of the validation data {validation.shape}")
    return training, validation


def get_data_splits_by_date_block(logger, df, train_months, validation_months):
    """
    Returns training & holdout set from the train data based on months represented
    by date_block (competition Specific)

    train_months: List of date_block numbers representing months.

    test_months: List of date_block numbers representing months.
    """
    logger.info("Splitting the data into train and holdout based on months...")
    logger.info(f"Training months {train_months}")
    logger.info(f"Validation months {validation_months}")
    training = df[df.date_block_num.isin(train_months)]
    validation = df[df.date_block_num.isin(validation_months)]
    logger.info(f"Shape of the training data {training.shape} ")
    logger.info(f"Shape of the validation data {validation.shape}")
    return training, validation


def create_stratify_labels_for_regression(
    df, target_feature="target",
):
    """Returns a new column to the tarining data generated based on the
    distribution of the continous target variable. This new column can be
    passed to the StratifiedKFold for stratified cv on top of continous
    target
    """
    df = df.reset_index(drop=False)
    df = df.sample(frac=1).reset_index(drop=True)
    num_bins = np.int(np.floor(1 + np.log2(len(df))))
    df.loc[:, "bins"] = pd.cut(df["target"], bins=num_bins, labels=False)
    # This portion of the code may be specific to current competition
    df = df.set_index("id")
    df = df.sort_index()
    return df
