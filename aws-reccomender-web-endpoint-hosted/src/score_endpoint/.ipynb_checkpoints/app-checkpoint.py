
# -*- coding: utf-8 -*-

'''
A basic Flask application for scoring web endpoint.
'''

import os
from collections import defaultdict
import pandas as pd
from surprise import dump

import flask
from flask import Flask, jsonify, Response
import warnings
warnings.filterwarnings("ignore")

#create an instance
app = Flask(__name__)

def locate_model(dest):
    
    """ 
    return path of binary model
    args:
       dest: folder for searching
    returns:
       model_path
    """

    for dirpath, dirnames, filenames in os.walk(dest):
        for filename in [f for f in filenames if f.endswith((".pkl", ".pickle"))]:
            model_path = os.path.join(dirpath, filename)
            return model_path
    return None

def model_reader(model_path):
    """ 
    return predictions and model class
    args:
       model_path: pickle file path
    returns:
       predictions and model
    """
    predictions, algo = dump.load(model_path)
    return predictions, algo

def get_top(predictions, n=10):
    
    '''
    Returns the the top-N recommendation from a set of predictions
    args:
       predictions: predictions generated in testing phase
       n: number of items to suggest (default=10)
    return:
       top_n dictionary: user-prediction dictionaries
    '''
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
        
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
        
    return top_n

def get_top_n_ui(top, uid):
    '''
    Returns the list of selected items
    args:
       top: user-prediction dictionaries
       uid: user id for filtering
    return:
       top_n dictionary: user-prediction dictionaries
    '''
    try:
        top_n_ui = [[iid for (iid, _) in user_ratings] for UID, user_ratings in top.items() if UID==uid][0]
        return top_n_ui
    except ValueError: 
        return 0

@app.route('/', methods=['GET'])
def health_checker():
    text = 'The Scoring application is ready!'
    return Response(text, status=200, mimetype='text/plain')

@app.route('/predict', methods=['GET','POST'])
def predictor():

    #Intiate variables
    data = defaultdict()
    data["success"] = False
    params = flask.request.args
    
    if 'uid' in params.keys():
        uid_toscore = str(params.get('uid'))
        model_path = locate_model(os.getcwd())
        predictions, _ = model_reader(model_path)
        uid_predictions = get_top_n_ui(get_top(predictions), uid_toscore)
        
        prediction_rank_lenght = len(uid_predictions)
        prediction_rank_labels = ["".join([" Product", str(i)]) for i in range(1,prediction_rank_lenght)]
        products_recommended = pd.DataFrame(list(zip(prediction_rank_labels, uid_predictions)), columns=['Product_Rank', 'Product_id'])

        data['response'] = products_recommended.to_dict()
        data['success'] = True
    
    return flask.jsonify(data)
