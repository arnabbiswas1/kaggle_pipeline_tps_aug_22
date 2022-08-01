"""This script changes the data types, creates parquet files.
Final output is written to the specified directory

Sample Usage:
    <PROJECT_HOME>$ python -m src.scripts.data_processing.process_raw_data
"""
import numpy as np
import pandas as pd

import src.munging.process_data_util as process_data
from src.common import com_util as util
from src.config import constants as constants

if __name__ == "__main__":
    # Create a Stream only logger
    logger = util.get_logger("process_raw_data")
    logger.info("Starting to process raw data")

    train_df, test_df, sample_submission_df = process_data.read_raw_data(
        logger,
        constants.RAW_DATA_DIR,
        index_col_name="id",
        train=True,
        test=True,
        sample_submission=True,
    )

    TARGET = "failure"

    target = train_df[TARGET]

    combined_df = pd.concat([train_df.drop([TARGET], axis=1), test_df])

    logger.info("Changing data type of target data ..")
    target = target.astype(np.int32)

    train_df = combined_df.iloc[0: len(train_df), :]
    # Make sure to use the name of the target below
    train_df = train_df.assign(failure=target)
    test_df = combined_df.iloc[len(train_df):, :]

    logger.info("Changing data type of submission  data ..")
    sample_submission_df = process_data.change_dtype(
        logger, sample_submission_df, np.float64, np.float32
    )
    logger.info(f"Writing processed feather files to {constants.PROCESSED_DATA_DIR}")
    train_df.to_parquet(
        f"{constants.PROCESSED_DATA_DIR}/train_processed.parquet", index=True
    )
    test_df.to_parquet(
        f"{constants.PROCESSED_DATA_DIR}/test_processed.parquet", index=True
    )
    sample_submission_df.to_parquet(
        f"{constants.PROCESSED_DATA_DIR}/sub_processed.parquet", index=True
    )

    logger.info("Raw data processing completed")
