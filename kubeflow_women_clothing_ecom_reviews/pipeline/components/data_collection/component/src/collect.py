#!/usr/bin/env python3

#
# Description
#

# Libraries ------------------------------------------------------------------------------------------------------------

## General
import os
import logging

## Extract
import kaggle
import pandas as pd

## Transform
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

## Load
from tensorflow.io import gfile


# DataCollector --------------------------------------------------------------------------------------------------------
class DataCollector:

    def __init__(self, config):
        self.dataset = config['dataset']
        self.raw_path = config['raw_path']
        self.raw_data = config['raw_data']
        self.variables = config['variables']
        self.target = config['target']
        self.random_state = config['random_state']
        self.test_size = config['test_size']
        self.val_size = config['val_size']
        self.interim_path = config['interim_path']
        self.df_names = config['interim_data']


    def extract(self, mode, bucket):
        logging.info('Initiating Data Extraction Processing...')
        kaggle.api.authenticate()
        logging.info('Fetching the data...')
        kaggle.api.dataset_download_files(dataset=self.dataset,
                                              path=self.raw_path,
                                              unzip=True)
        raw_df = pd.read_csv(os.path.join(self.raw_path, self.raw_data))
        if mode == 'cloud':
            out_csv_gcs = f'{bucket}/{self.raw_path}/{self.raw_data}'
            logging.info(f'Data extracted under {bucket}/{self.raw_path}...')
            with gfile.GFile(name=out_csv_gcs, mode='w') as file:
                raw_df.to_csv(file)
        return raw_df

    def transform(self, raw_df):
        inter_data = raw_df.copy()
        logging.info('Initiating Data Processing...')
        logging.info('Reformat column names...')
        columns_formatted = [col.lower().replace(" ", "_") for col in inter_data.columns]
        inter_data.columns = columns_formatted

        logging.info('Select column for classification...')
        inter_data = inter_data[self.variables + [self.target]]
        logging.info('Handling with missing data...')
        inter_data = inter_data[~inter_data[self.variables[1]].isnull()]

        logging.info('Oversampling...')
        y = inter_data[self.target]
        x = inter_data[inter_data.columns.difference([self.target])]
        ros = RandomOverSampler(random_state=self.random_state)
        x_over, y_over = ros.fit_resample(x, y)

        logging.info('Train, Validation, Test splitting...')
        x_train, x_test, y_train, y_test = train_test_split(x_over, y_over, test_size=self.test_size,
                                                            random_state=self.random_state)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size,
                                                          random_state=self.random_state)
        logging.info('Data processing successfully completed!')
        return x_train, x_test, x_val, y_train, y_test, y_val

    def load(self, x_train, x_test, x_val, y_train, y_test, y_val,
             mode, bucket):

        logging.info('Initiating Data Loading...')
        x_dfs = [x_train, x_test, x_val]
        y_dfs = [y_train, y_test, y_val]
        dfs = [pd.merge(x_df, y_df, how="left", left_index=True, right_index=True) for x_df, y_df in zip(x_dfs, y_dfs)]

        if mode == 'cloud':
            out_gcs = []
            for df_name, df in zip(self.df_names, dfs):
                out_csv_gcs = f'{bucket}/{self.interim_path}/{df_name}'
                logging.info(f'Loading data to {out_csv_gcs}...')
                with gfile.GFile(name=out_csv_gcs, mode='w') as file:
                    df.to_csv(file, index=False)
                out_gcs.append(out_csv_gcs)
                logging.info(f'Data successfully loaded under {out_csv_gcs}')
            return tuple(out_gcs)

        else:
            out_path = []
            os.mkdir(path=self.interim_path)
            for df_name, df in zip(self.df_names, dfs):
                out_csv_path = os.path.join(self.interim_path, df_name)
                logging.info(f'Loading data to {out_csv_path}...')
                df.to_csv(out_csv_path, index=False)
                logging.info(f'{out_csv_path} successfully loaded!')
                out_path.append(out_csv_path)
            return tuple(out_path)
