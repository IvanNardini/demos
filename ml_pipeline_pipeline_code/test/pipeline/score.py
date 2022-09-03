#Data
import pandas as pd

#Preprocessing
from sklearn.model_selection import train_test_split

#Evaluate Pipeline
from postprocessing import PostProcessing

#Utils
import logging
import time
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

def score(pipe_path, input_data):
    
    pipeline = PostProcessing.load(pipe_path)
    predictions = pipeline.predict(input_data)

    return pipeline, predictions
   
if __name__ == '__main__':

  # Read configuration
    stream = open('config.yaml', 'r')
    config = yaml.load(stream)

    DATA_INGESTION = config['data_ingestion']
    PREPROCESSING = config['preprocessing']
    FEATURES_ENGINEERING = config['features_engineering']
    PIPE_TRAINING = config['pipeline_training']

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
    
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Scoring process started!')
    start = time.time()
    pipeline, predictions = score(PIPE_TRAINING['pipe_path'], X_test)
    end = time.time()
    duration = end - start
    logging.info('Scoring process successfully completed!')
    logging.info('Time for Scoring: {} seconds!'.format(round(duration, 5)))

    ##################################################################################

    #Evaluate
    print("*"*20)
    print("Model Predictions".center(20, '*'))
    print("*"*20)
    print()
    print('First 10 prediticions are: {}'.format(predictions[:10]))
    print()
  
    print("*"*20)
    print("Model Assessment".center(20, '*'))
    print("*"*20)
    PostProcessing.evaluate_classification(X_test, y_test, pipeline, predictions)
