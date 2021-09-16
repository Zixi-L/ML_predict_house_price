"""
Short description - This module contains code to process and prepare data for ML
modelling

:author: Zixi Luo
"""

import argparse
import os
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def main():
    """
    take the file and test_size from command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='data/home_data.csv', 
                        help='pass the file path, and the default path is data/home_data.csv')
    parser.add_argument("--test_size", type=float, default=0.33, 
                        help='take the test_size, and the default value is 0.33')
    
    
    args = parser.parse_args()
    file = args.file
    test_size = args.test_size

def read_file(file):
    """
    read the fild
    """
    df = pd.read_csv(file)
    return df

def preprocessing(file, test_size):
    """
    preprocessing the csv file: 
    onehotencoder the zipcode and split the train, test size
    """
    # get the datafram
    df = read_file(file)
    
    # drop idm date and split the dataframe into x and y
    df = df.drop(['id','date'], axis=1)
    x = df.drop(['price'], axis=1)
    y = df['price']
    
    # onehotencoder zipcode and drop zipcode from x
    enc = OneHotEncoder(handle_unknown='ignore')
    zipcode_df =  pd.DataFrame(enc.fit_transform(x[['zipcode']]).toarray())
    x = x.join(zipcode_df)
    x = x.drop(['zipcode'], axis = 1)
    
    # split train and test data set
    x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=test_size)
    
    return x_train, x_val, y_train, y_val
    
    
if __name__ == "__main__":
    main()