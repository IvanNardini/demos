
'''
This module runs the entire Dash Application

'''

import dash_table
import dash_core_components as component
import dash_html_components as html
from dash.dependencies import Input, Output

from app import *
from layouts import c_id_div, predictions_div
import callbacks


app.layout= html.Div(
    
    id="score_gui", children=[
        html.H1('Recommendation System for Purchase Data'),
        html.H2('Scoring Interactive Web Service (Basic Concept)'),
        html.Br(), 
        html.Div(children=[c_id_div, predictions_div], className="row")
    ])

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=True)
