#!/usr/bin/env python3

#
# Description
#

# Libraries ------------------------------------------------------------------------------------------------------------

import logging
import pandas as pd
from tensorflow.io import gfile


# Helpers --------------------------------------------------------------------------------------------------------------

def load_data(input_path, input_data, mode, bucket):
    if mode == 'cloud':
        input_gcs = f'{bucket}/{input_path}/{input_data}'
        logging.info(f'Loading data from {input_gcs}...')
        with gfile.GFile(name=input_gcs, mode='r') as file:
            df = pd.read_csv(file)
    else:
        input_file = f'{input_path}/{input_data}'
        df = pd.read_csv(input_file)
    logging.info(f'{input_data} successfully loaded!')
    return df
