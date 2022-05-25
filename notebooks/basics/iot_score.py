# This script generates the scoring file
# with the init and run functions needed to 
# operationalize the anomaly detection sample

import json
import pandas
import joblib
from azureml.core.model import Model

def init():
    global model
    # this is a different behavior than before when the code is run locally, even though the code is the same.
    model_path = Model.get_model_path('lgb2.pkl')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

# note you can pass in multiple rows for scoring
def run(input_str):
    try:
        input_json = json.loads(input_str)
        input_df = pandas.DataFrame([[input_json['sepal']['length'],input_json['sepal']['width'],input_json['petal']['length'],input_json['petal']['width']]])
        pred = model.predict(input_df)
        print("Prediction is ", pred[0])
    except Exception as e:
        result = str(e)
        
    if pred[0] == 0:
        input_json['species']='0-Iris-setosa'
    elif pred[0] == 1:
        input_json['species']='1-Iris-versicolor'
    else:
        input_json['species']='2-Iris-virginica'
        
    return [json.dumps(input_json)]
