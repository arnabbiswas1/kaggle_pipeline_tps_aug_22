import pandas as pd

from sklearn.preprocessing import LabelEncoder

import src.munging as process_data
import src.common as common
import src.config.constants as constants


def main():
    # Create a Stream only logger
    logger = common.get_logger("reduce_memory")
    logger.info("Starting to Reduce Memory")

    TARGET = "claim"
    FEATURE_FILE_NAME = "all_combined.parquet"

    train_df, test_df, _ = process_data.read_processed_data(
        logger,
        constants.PROCESSED_DATA_DIR,
        train=True,
        test=True,
        sample_submission=True,
    )

    combined_df = pd.concat([train_df.drop(TARGET, axis=1), test_df])

    del train_df, test_df
    common.trigger_gc(logger)

    combined_df_reduced = process_data.reduce_memory_usage(
        logger, combined_df, verbose=True
    )
    logger.info(
        f"If DF equals to DF before size reduction {combined_df_reduced.equals(combined_df)}"
    )

    del combined_df
    common.trigger_gc(logger)

    feature_file_list = [
        "features_fenc_cat_intrc.parquet",
        "features_power_1.parquet",
        "features_power_2.parquet",
        "features_bin_cut.parquet",
        "features_bin_qcut.parquet",
        "features_row_wise_stat_extra.parquet",
    ]

    for name in feature_file_list:
        logger.info(f"Compressing {name}")
        df = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/{name}")

        logger.info(f"Shape before merging {df.shape}")

        df_reduced = process_data.reduce_memory_usage(logger, df, verbose=True)
        logger.info(f"If DF equals to DF before size reduction {df_reduced.equals(df)}")

        del df
        common.trigger_gc(logger)

        combined_df_reduced = pd.concat([combined_df_reduced, df_reduced], axis=1)
        logger.info(f"Shape after merging {combined_df_reduced.shape}")

        del df_reduced
        common.trigger_gc(logger=logger)

    for name in combined_df_reduced.select_dtypes("object"):
        logger.info(f"Label encoding {name}")
        lb = LabelEncoder()
        combined_df_reduced[name] = lb.fit_transform(combined_df_reduced[name])

    combined_df_reduced = process_data.reduce_memory_usage(
        logger=logger, df=combined_df_reduced, verbose=True
    )

    logger.info(
        f"Writing generated features to {constants.FEATURES_DATA_DIR}/{FEATURE_FILE_NAME}"
    )
    combined_df_reduced.to_parquet(
        f"{constants.FEATURES_DATA_DIR}/{FEATURE_FILE_NAME}", index=True
    )


if __name__ == "__main__":
    main()
