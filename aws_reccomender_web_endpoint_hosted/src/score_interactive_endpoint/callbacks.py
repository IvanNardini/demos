
'''
This module contains the predict callback 

'''
from dash.dependencies import Input, Output

from app import *

@app.callback([Output('table', component_property='columns'), Output('table', component_property='data')],[Input(component_id='uid', component_property='value')])
def predict(uid):
    columns = []
    products_recommended = []
    if uid:
        model_path = locate_model(os.getcwd())
        predictions, _ = model_reader(model_path)
        uid_predictions = get_top_n_ui(get_top(predictions), uid)
        prediction_rank_lenght = len(uid_predictions)
        prediction_rank_labels = ["".join([" Product", str(i)]) for i in range(1,prediction_rank_lenght)]
        products_recommended = pd.DataFrame(list(zip(prediction_rank_labels, uid_predictions)), columns=['Product_Rank', 'Product_id'])
        columns=[{"name": i, "id": i} for i in products_recommended.columns]
        return columns, products_recommended.to_dict('records')
    else:
        return columns, products_recommended
