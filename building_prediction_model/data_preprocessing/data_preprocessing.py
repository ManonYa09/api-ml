import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import sys

# Adjust the path to go up two directories to the project root
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PACKAGE_ROOT))

# Import config from building_prediction_model
from building_prediction_model.config import config

# Transformer to drop specified columns
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        # If variables_to_drop is None, default to config.FEATURES_DROP
        self.variables_to_drop = variables_to_drop or config.FEATURES_DROP
    
    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self
    
    def transform(self, X):
        # Drop the specified columns from the DataFrame
        X = X.drop(columns=self.variables_to_drop)
        return X

# Transformer to encode and create dummy variables
class EncodeAndBind(BaseEstimator, TransformerMixin):
    def __init__(self, encode=None, dummy=None):
        self.encode = encode
        self.dummy = dummy
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Replace 'Male' with 0 and 'Female' with 1
        X[self.encode].replace({'Male': 0, 'Female': 1}, inplace=True)
        # Create dummy variables
        X = pd.get_dummies(X, columns=[self.dummy])
        # Replace boolean values with 1 and 0
        X.replace({True: 1, False: 0}, inplace=True)
        return X

# Transformer to normalize specified variables
class Scale(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables or config.FEATURES_TO_SCALE
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Scale each variable to range 0-1
        for variable in self.variables:
            X[variable] = (X[variable] - X[variable].min()) / (X[variable].max() - X[variable].min())
        return X