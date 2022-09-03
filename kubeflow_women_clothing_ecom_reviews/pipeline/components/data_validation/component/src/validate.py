#!/usr/bin/env python3

#
# Description
#

# Libraries ------------------------------------------------------------------------------------------------------------

## General
import os
import logging

## Validate
import pandera as pdv
from pandera import Column, Check, DataFrameSchema
import pandas as pd
import numpy as np


# DataValidator --------------------------------------------------------------------------------------------------------

class DataValidator:

    def __init__(self, config):
        self.raw_path = config['raw_path']
        self.raw_data = config['raw_data']
        self.random_state = config['random_state']
        self.sample_size = config['sample_size']
        self.columns = config['columns']

        # Define checks
        self.check_ge_min = Check(lambda s: s >= 0)
        self.check_le_max = Check(lambda s: s <= max(s))

    def validate(self, df):
        logging.info('Defining schema...')
        schema_content = {}
        for col in self.columns:
            if pd.api.types.is_int64_dtype(np.array(df[col])):
                schema_content[col] = Column(pdv.Int, checks=[self.check_ge_min, self.check_le_max])
            else:
                schema_content[col] = Column(pdv.String, nullable=True)
        schema = DataFrameSchema(schema_content)
        logging.info('Generating a sample for validation...')
        val_sample = df[self.columns].sample(n=self.sample_size, random_state=self.random_state)
        logging.info('Validating...')
        val_df = schema.validate(val_sample)
        print(val_df.head(5))
        validation_status = val_df.empty
        return validation_status

    @staticmethod
    def check_validity(validation_status):
        if not validation_status:
            logging.info('Data Validation successfully completed!')
