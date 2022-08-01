import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-aug-2022"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_lgb_baseline_0801_1244_0.58406.gz"
SUBMISSION_MESSAGE = '"LGB benchmark Startified KFold-5"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
