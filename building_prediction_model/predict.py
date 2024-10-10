import pandas as pd
import numpy as np
import joblib
import warnings 
warnings.filterwarnings("ignore")
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from building_prediction_model.config import config
from building_prediction_model.data_preprocessing.data_handling import load_dataset, load_pipeline, separate_data

features = {
    
    "RowNumber": 1223,
    "CustomerId": 15600700 ,
    "Surname": "Pan",
    "CreditScore": 890,
    "Geography": "France",
    "Gender": "Male",
    "Age": 19,
    "Tenure": 1,
    "Balance": 1000,
    "NumOfProducts": 2,
    "HasCrCard": 2,
    "IsActiveMember": 2,
    "EstimatedSalary": 350
}

def generate_predictions(features):
    X = pd.DataFrame([features])
    # X, y = separate_data(test_data)
    pred1 = load_pipeline('classification.pkl').predict(X)
    # pred1 = np.where(pred1 == 1, 'Churn', 'Not Churn')
    
    # Print or return the predicted outputs
    # print("Predicted output: ", pred1, len(pred1))
    return  pred1[0]
if __name__ == '__main__':
    print(generate_predictions(features))