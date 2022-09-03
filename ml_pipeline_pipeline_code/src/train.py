# Read data
import pandas as pd

#Preprocessing
from sklearn.model_selection import train_test_split

#Pipeline
from pipeline import pipeline

#Model
from postprocessing import PostProcessing

#Utils
import logging
import time
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

def train():
    # Read Data
    data = pd.read_csv(DATA_INGESTION['data_path'])
    target = DATA_INGESTION['data_map']['target']
    variables = DATA_INGESTION['data_map']['variables']

    #Preprocessing
    flt = data['umbrella_limit']>=0
    data = data[flt]
    data[target] = data[target].map(FEATURES_ENGINEERING['target_encoding'])

    #Split data
    X_train, X_test, y_train, y_test = train_test_split(data[variables], data[target],
                                        test_size=PREPROCESSING['train_test_split_params']['test_size'],
                                        random_state=PREPROCESSING['train_test_split_params']['random_state'])    
    #Train Pipeline
    Pipeline_Fit = pipeline.fit(X_train, y_train)

    #Save Model
    PostProcessing.save(Pipeline_Fit, PIPE_TRAINING['pipe_path'])

if __name__ == '__main__':

    # Read configuration
    stream = open('config.yaml', 'r')
    config = yaml.load(stream)

    DATA_INGESTION = config['data_ingestion']
    PREPROCESSING = config['preprocessing']
    FEATURES_ENGINEERING = config['features_engineering']
    PIPE_TRAINING = config['pipeline_training']

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Training process started!')
    start = time.time()
    train()
    end = time.time()
    duration = end - start
    logging.info('Training process successfully completed!')
    logging.info('Time for training: {} seconds!'.format(round(duration, 5)))