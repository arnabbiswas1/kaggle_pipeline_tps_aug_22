# Kaggle Pipeline for **Kaggle TPS August 2022**

## Steps to execute:

1. Clone the source code from github under <PROJECT_HOME> directory.

        > git clone https://github.com/arnabbiswas1/k_tab_aug_22.git

    This will create the following directory structure:
    
        > <PROJECT_HOME>/k_tab_aug_22

2. Create conda env:

        > conda env create --file environment.yml

3. Go to `<PROJECT_HOME>/k_tab_aug_22` and activate conda environment:

        > conda activate py_k

3. Go to the raw data directory at `<PROJECT_HOME>/k_tab_aug_22/data/raw`. Download dataset from Kaggle (Kaggle API should be configured following [link](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication)):

        > kaggle competitions download -c tabular-playground-series-aug-2022

4. Unzip the data:

        > unzip tabular-playground-series-aug-2022.zip

5. Set the value of variable `HOME_DIR` at `<PROJECT_HOME>/k_tab_aug_22/src/config/constants.py` with the absolute path of `<PROJECT_HOME>/k_tab_aug_22`

6. To process raw data into parquet format, go to `<PROJECT_HOME>/k_tab_aug_22`. Execute the following:

        > python -m src.scripts.data_processing.process_raw_data

    This will create 3 parquet files under `<PROJECT_HOME>/k_tab_aug_22/data/processed` representing train, test and sample_submission CSVs

7. To trigger feature engineering, go to `<PROJECT_HOME>/k_tab_aug_22`. Execute the following:

        > python -m src.scripts.data_processing.create_features

   This will create a parquet file containing all the engineered features under `<PROJECT_HOME>/k_tab_aug_22/data/features`

8. To train the baseline model with LGBM, `<PROJECT_HOME>/k_tab_aug_22`. Execute the following:

        > python -m src.scripts.training.lgb_baseline

     This will create the submission file under `<PROJECT_HOME>/k_tab_aug_22/submissions`. Out of Fold predictions under `<PROJECT_HOME>/k_tab_aug_22/oof` and CSVs capturing feature importances under `<PROJECT_HOME>/k_tab_aug_22/fi`

Result of the experiment will be tracked at `<PROJECT_HOME>/k_tab_aug_22/tracking/tracking.csv`

9. To submit the submission file to kaggle, go to `<PROJECT_HOME>/k_tab_aug_22/submissions`:

        > python -m submissions_1.py

## Note:

Following is needed for visualizing plots for optuna using plotly (i.e. plotly dependency):

> jupyter labextension install jupyterlab-plotly@4.14.3
