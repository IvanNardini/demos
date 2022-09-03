'''
Compile pipeline contains the pipeline object
'''
import data_preprocessing as Data_Prep
import feature_engineering as Feat_Eng
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

#Utils
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

# Read configuration
stream = open('config.yaml', 'r')
config = yaml.load(stream)

DATA_INGESTION = config['data_ingestion']
PREPROCESSING = config['preprocessing']
FEATURES_ENGINEERING = config['features_engineering']
PIPE_TRAINING = config['pipeline_training']

pipeline = Pipeline(
    [
        ('Dropper', Data_Prep.Dropper(dropped_columns=PREPROCESSING['dropped_columns'])),

        ('Renamer', Data_Prep.Renamer(renamed_columns=PREPROCESSING['renamed_columns'])),

        ('Missing_Imputer', Data_Prep.Missing_Imputer(missing_predictors=PREPROCESSING['missing_predictors'], 
                                                    replace='missing')), 

        ('Binner', Feat_Eng.Binner(binning_meta=FEATURES_ENGINEERING['binning_meta'])), 

        ('Encoder', Feat_Eng.Encoder(encoding_meta=FEATURES_ENGINEERING['encoding_meta'])),

        ('Dumminizer', Feat_Eng.Dumminizer(columns_to_dummies=FEATURES_ENGINEERING['nominal_predictors'])), 

        ('Scaler', Feat_Eng.Scaler(features=FEATURES_ENGINEERING['features'])),

        ('Feature_selector', Feat_Eng.Feature_selector(features_selected=FEATURES_ENGINEERING['features_selected'])), 

        ('SMOTE', SMOTE(random_state=FEATURES_ENGINEERING['random_sample_smote'])), 

        ('RandomForestClassifier', RandomForestClassifier(max_depth=PIPE_TRAINING['RandomForestClassifier']['max_depth'], 
                                                        min_samples_split=PIPE_TRAINING['RandomForestClassifier']['min_samples_split'], 
                                                        n_estimators=PIPE_TRAINING['RandomForestClassifier']['n_estimators'], 
                                                        random_state=PIPE_TRAINING['RandomForestClassifier']['random_state']))

    ]
)