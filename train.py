"""
Short description - This module contains code to train the model and save it.

:author: Zixi Luo
"""

from preprocess import * 

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from joblib import dump, load

from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import RandomizedSearchCV

import numpy as np

def train(x_train, y_train, n_estimators, max_features, max_depth):
    rf = RandomForestRegressor(n_estimators=n_estimators,
                              max_features=max_features,
                              max_depth=max_depth)
    rf.fit(x_train, y_train)
    return rf

def save_model(model,output_path):
    dump(model, f"{output_path}/trained_model.joblib") 
    
def main():
    """
    take the file and test_size from command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='data/home_data.csv', 
                        help='pass the file path, and the default path is data/home_data.csv')
    parser.add_argument("--test_size", type=float, default=0.33, 
                        help='take the test_size, and the default value is 0.33')
    
    parser.add_argument("--n_estimators", type=int, default=90, 
                        help='take the n_estimators, and the default value is 90')
   
    parser.add_argument("--max_features", type=int, default=80, 
                        help='take the max_features, and the default value is 80')
   
    parser.add_argument("--max_depth", type=int, default=20, 
                        help='take the max_depth, and the default value is 20')

    args = parser.parse_args()
    file = args.file
    test_size = args.test_size
    n_estimators = args.n_estimators,
    max_features = args.max_features,
    max_depth = args.max_depth
    
    x_train, x_val, y_train, y_val = preprocessing(file, test_size)
    
    train(x_train, y_train, n_estimators, max_features, max_depth)
    
    
    
if __name__ == "__main__":
    main()
    
