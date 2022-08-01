"""
LGB benchmark StratifiedKFold-5
"""

import os
from timeit import default_timer as timer
from datetime import datetime

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import src.common as common
import src.config.constants as constants
import src.munging as process_data
import src.modeling as model

common.set_timezone()
start = timer()

# Create RUN_ID
RUN_ID = datetime.now().strftime("%m%d_%H%M")
MODEL_NAME = os.path.basename(__file__).split(".")[0]

SEED = 42
EXP_DETAILS = "LGB benchmark Stratified KFold-5"

TARGET = "failure"
N_SPLIT = 5

MODEL_TYPE = "lgb"
OBJECTIVE = "binary"
BOOSTING_TYPE = "gbdt"
METRIC = "auc"
VERBOSE = 100
N_THREADS = -1
NUM_LEAVES = 31
MAX_DEPTH = -1
N_ESTIMATORS = 10000
LEARNING_RATE = 0.1
EARLY_STOPPING_ROUNDS = 100


lgb_params = {
    "objective": OBJECTIVE,
    "boosting_type": BOOSTING_TYPE,
    "learning_rate": LEARNING_RATE,
    "num_leaves": NUM_LEAVES,
    "tree_learner": "serial",
    "n_jobs": N_THREADS,
    "seed": SEED,
    "max_depth": MAX_DEPTH,
    "max_bin": 255,
    "metric": METRIC,
    "verbose": -1,
}


LOGGER_NAME = "training"
logger = common.get_logger(LOGGER_NAME, MODEL_NAME, RUN_ID, constants.LOG_DIR)
common.set_seed(SEED)
logger.info(f"Running for Model Number [{MODEL_NAME}] & [{RUN_ID}]")

common.update_tracking(RUN_ID, "model_number", MODEL_NAME, drop_incomplete_rows=True)
common.update_tracking(RUN_ID, "model_type", MODEL_TYPE)
common.update_tracking(RUN_ID, "metric", "roc_auc")
common.update_tracking(RUN_ID, "n_estimators", N_ESTIMATORS)
common.update_tracking(RUN_ID, "learning_rate", LEARNING_RATE)
common.update_tracking(RUN_ID, "num_leaves", NUM_LEAVES)
common.update_tracking(RUN_ID, "early_stopping_rounds", EARLY_STOPPING_ROUNDS)

train_df, test_df, sample_submission_df = process_data.read_processed_data(
    logger,
    constants.PROCESSED_DATA_DIR,
    train=True,
    test=True,
    sample_submission=True,
)

cat_features = [
    "product_code", "attribute_0", "attribute_1"
]

combined_df = pd.concat([train_df.drop(TARGET, axis=1), test_df])

for name in cat_features:
    lb = LabelEncoder()
    combined_df[name] = lb.fit_transform(combined_df[name])

target = train_df[TARGET]

train_df = combined_df.loc[train_df.index]
train_df[TARGET] = target

test_df = combined_df.loc[test_df.index]

train_X = train_df.drop([TARGET], axis=1)
train_Y = train_df[TARGET]
test_X = test_df

logger.info(
    f"Shape of train_X : {train_X.shape}, test_X: {test_X.shape}, train_Y: {train_Y.shape}"
)

predictors = list(train_X.columns)
kfold = StratifiedKFold(n_splits=N_SPLIT, random_state=SEED, shuffle=True)

common.update_tracking(RUN_ID, "no_of_features", len(predictors), is_integer=True)
common.update_tracking(RUN_ID, "cv_method", "KFold")
common.update_tracking(RUN_ID, "n_fold", N_SPLIT)


results_dict = model.lgb_train_validate_on_cv(
    logger,
    run_id=RUN_ID,
    train_X=train_X,
    train_Y=train_Y,
    test_X=test_X,
    metric="roc_auc",
    kf=kfold,
    features=predictors,
    params=lgb_params,
    n_estimators=N_ESTIMATORS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    cat_features="auto",
    verbose_eval=100,
)

common.update_tracking(RUN_ID, "lb_score", 0, is_integer=True)

train_index = train_df.index

common.save_artifacts(
    logger,
    target=TARGET,
    is_plot_fi=True,
    result_dict=results_dict,
    submission_df=sample_submission_df,
    train_index=train_index,
    model_number=MODEL_NAME,
    run_id=RUN_ID,
    sub_dir=constants.SUBMISSION_DIR,
    oof_dir=constants.OOF_DIR,
    fi_dir=constants.FI_DIR,
    fi_fig_dir=constants.FI_FIG_DIR,
)

end = timer()
common.update_tracking(RUN_ID, "training_time", end - start, is_integer=True)
common.update_tracking(RUN_ID, "comments", EXP_DETAILS)
logger.info("Execution Complete")
