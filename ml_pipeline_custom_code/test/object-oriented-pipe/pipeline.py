'''
pipeline modules contains the pipeline object
'''
from templates.data_preprocessing import Preprocessing
from templates.features_engineering import FeatureEngineering
from templates.modelling import Models
from templates.postprocessing import PostProcessing

#Utils
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


class Pipeline():   
    
    def __init__(self, dropped_columns, renamed_columns, missing_predictors, 
                 target, predictors, nominal_predictors, features, features_selected,
                 target_encoding, binning_meta, encoding_meta):

        ##Data
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        ##Declared Variables
        self.dropped_columns = dropped_columns
        self.renamed_columns = renamed_columns
        self.anomalies = 'umbrella_limit'
        self.missing_predictors = missing_predictors
        self.target = target
        self.predictors = predictors
        self.nominal_predictors = nominal_predictors
        self.features = features
        self.features_selected = features_selected
        self.target_encoding = target_encoding
        self.binning_meta = binning_meta
        self.encoding_meta = encoding_meta

        ## metadata
        self.replace = 'missing'
        self.test_size = 0.10
        self.random_state_sample = 1
        self.scaler = None
        self.random_state_smote = 9
        self.random_state_model = 8
        self.max_depth = 25
        self.min_samples_split = 5
        self.n_estimators= 300

        #Model
        self.model = None

    # =====================================================================================================

    #fit pipeline
    def fit(self, data):

        #Initialize
        self.data = data
        #Step1: Drop columns 
        self.data = Preprocessing.dropper(self, self.data, self.dropped_columns)
        #Step2: Rename columns 
        self.data = Preprocessing.renamer(self, self.data, self.renamed_columns)
        #Step3: Remove anomalies
        self.data = Preprocessing.anomalizier(self, self.data, self.anomalies)
        #Step4: Impute missing
        self.data = Preprocessing.missing_imputer(self, self.data, 
                                                  self.missing_predictors,
                                                  replace=self.replace)
        #Step6: Split data
        self.X_train, self.X_test, self.y_train, self.y_test = Preprocessing.data_splitter(
                                                    self,
                                                    self.data,
                                                    self.target,
                                                    self.predictors,
                                                    self.test_size,
                                                    self.random_state_sample
                                                    )
        #Step7: Encode Target
        self.y_train = FeatureEngineering.target_encoder(self, self.y_train, 
                                                        self.target_encoding)
        self.y_test = FeatureEngineering.target_encoder(self, self.y_test, 
                                                        self.target_encoding)

        #Step8: Bin variables
        self.X_train = FeatureEngineering.binner(self, self.X_train,
                                                        self.binning_meta)
        self.X_test = FeatureEngineering.binner(self, self.X_test,
                                                        self.binning_meta)
        #Step9: Encoding Variables
        self.X_train = FeatureEngineering.encoder(self, self.X_train, 
                                                        self.encoding_meta)
        self.X_test = FeatureEngineering.encoder(self, self.X_test, 
                                                        self.encoding_meta)
        #Step10: Generate Dummies
        self.X_train = FeatureEngineering.dumminizer(self, self.X_train, 
                                                            self.nominal_predictors)
        self.X_test = FeatureEngineering.dumminizer(self, self.X_test, 
                                                            self.nominal_predictors)
        #Step11: Scale Features
        self.scaler = FeatureEngineering.scaler_trainer(self, self.X_train)
        self.X_train = FeatureEngineering.scaler_transformer(self, self.X_train, 
                                                            self.features,
                                                            self.scaler)
        self.X_test = FeatureEngineering.scaler_transformer(self, self.X_test, 
                                                            self.features,
                                                            self.scaler)
        #Step 12: Select Features
        self.X_train = FeatureEngineering.features_selector(self, self.X_train,
                                                            self.features_selected)

        self.X_test = FeatureEngineering.features_selector(self, self.X_test,
                                                            self.features_selected)
        #Step12: Balancing
        self.X_train, self.y_train = FeatureEngineering.balancer(self, self.X_train, 
                                                                self.y_train, 
                                                                self.random_state_smote)
        #Step13: Model Fit 
        self.model = Models.RFor(self, max_depth=self.max_depth, 
                        min_samples_split=self.min_samples_split, 
                        n_estimators=self.n_estimators, random_state=self.random_state_model)
        self.model.fit(self.X_train, self.y_train)

        return self

    #transform data
    def transform(self, data):
        #Initialize
        data = data.copy()
        #Step1: Drop columns 
        data = Preprocessing.dropper(self, data, self.dropped_columns)
        #Step2: Rename columns 
        data = Preprocessing.renamer(self, data, self.renamed_columns)
        #Step3: Remove anomalies
        data = Preprocessing.anomalizier(self, data, self.anomalies)
        #Step4: Impute missing
        data = Preprocessing.missing_imputer(self, data, 
                                                  self.missing_predictors,
                                                  replace=self.replace)
        #Step5: Bin variables
        data = FeatureEngineering.binner(self, data,
                                        self.binning_meta)
        #Step6: Encoding Variables
        data = FeatureEngineering.encoder(self, data, 
                                        self.encoding_meta)
        #Step7: Generate Dummies
        data = FeatureEngineering.dumminizer(self, data, 
                                            self.nominal_predictors)
        #Step8: Scale Features
        data = FeatureEngineering.scaler_transformer(self, data, 
                                                    self.features,
                                                    self.scaler)
        #Step 9: Select Features
        data = FeatureEngineering.features_selector(self, data,
                                                    self.features_selected)
        return data

    #predict
    def predict(self, data):
        #Step1: Engineer the data
        data = self.transform(data)
        #Step2: Predict
        predictions = self.model.predict(data)
        return predictions
        
    #evaluate
    def evaluate(self):
        PostProcessing.evaluate_classification(self, self.model, self.X_train, self.y_train, 
                                     self.X_test, self.y_test)