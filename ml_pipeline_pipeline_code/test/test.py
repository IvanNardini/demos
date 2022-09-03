#Read Data
import pandas as pd

#Read pipeline
import joblib

#Utils
import logging
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info('Training process started!')
    df = pd.read_csv(config['paths']['data_path'])
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