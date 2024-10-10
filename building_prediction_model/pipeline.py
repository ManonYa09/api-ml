from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys
import warnings 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
warnings.filterwarnings("ignore")
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from sklearn.linear_model import LogisticRegression

from building_prediction_model.config import config
import building_prediction_model.data_preprocessing.data_preprocessing as pp
import numpy as np 
# Drop unnecessary columns using a FunctionTransformer
def drop_columns(X):
    return X.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Define the pipeline
classification_pipeline = Pipeline(steps=[
    ('drop_columns', FunctionTransformer(drop_columns, validate=False)),
    ('preprocess', ColumnTransformer(transformers=[
        ('gender', OneHotEncoder(), ['Gender']),
        ('geo', OneHotEncoder(), ['Geography']),
        ('scale', MinMaxScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'])
    ], remainder='passthrough')),
    ('model', LogisticRegression(random_state=12))
])
