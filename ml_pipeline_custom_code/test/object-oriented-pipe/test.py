#Read Data
import pandas as pd

#Pipeline
from pipeline import *

#Utils
import logging
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

# Read configuration
stream = open('config.yaml', 'r')
config = yaml.load(stream)

DATA_INGESTION = config['data_ingestion']
PREPROCESSING = config['preprocessing']
FEATURES_ENGINEERING = config['features_engineering']
MODEL_TRAINING = config['model_training']
    

pipeline = Pipeline(
                    dropped_columns=PREPROCESSING['dropped_columns'],
                    renamed_columns=PREPROCESSING['renamed_columns'],
                    missing_predictors=PREPROCESSING['missing_predictors'],
                    target = DATA_INGESTION['data_map']['target'],
                    predictors = PREPROCESSING['predictors'],
                    nominal_predictors=FEATURES_ENGINEERING['nominal_predictors'],
                    features=FEATURES_ENGINEERING['features'], 
                    features_selected=FEATURES_ENGINEERING['features_selected'],
                    target_encoding = FEATURES_ENGINEERING['target_encoding'],
                    binning_meta = FEATURES_ENGINEERING['binning_meta'],
                    encoding_meta=FEATURES_ENGINEERING['encoding_meta']
                    )

if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Training process started!')
    df = pd.read_csv(DATA_INGESTION['data_path'])
    pipeline.fit(df)
    logging.info('Training process successfully completed!')

    print()    
    print("*"*20)
    print("Model Assessment".center(20, '*'))
    print("*"*20)
    pipeline.evaluate()
    print("*"*20)
    print("Model Predictions".center(20, '*'))
    print("*"*20)

    logging.info('Scoring process started!')
    predictions = pipeline.predict(df)
    logging.info('Scoring process successfully completed!')

    print()
    print('First 10 prediticions are: {}'.format(predictions[:10]))
    print()