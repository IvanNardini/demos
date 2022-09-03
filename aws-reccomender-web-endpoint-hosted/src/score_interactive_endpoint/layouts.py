
'''
This module contains the layout of the application
'''

import dash_table
import dash_core_components as component
import dash_html_components as html

#Customerid Input
c_id_component = component.Input(id="uid", 
                                 type="text", 
                                 placeholder='100')

c_id_div = html.Div(id='customerid', children=[html.H6("Please enter your customer ID : "), 
                                               c_id_component,
                                               html.Br()], 
                   className="two columns")

#Prediction Output
predictions = dash_table.DataTable(id='table',
                                   css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                                   style_as_list_view=True, 
                                   style_cell={'width': '30%', 'textAlign': 'left'},
                                   style_header={'backgroundColor': 'white', 'fontWeight': 'bold'})

predictions_div = html.Div(id='predictions', 
                           children=[html.H6(children='Products Recommended'),
                                     predictions],
                           style={'width': '25%'}, 
                           className="ten columns")
