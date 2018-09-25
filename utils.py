import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, Imputer, MinMaxScaler
from sklearn.model_selection import train_test_split

def label_encode(df):
    """
     Label encoder to encode the labels.
      param: Dataframe whose columns has to be encoded.
      return: Label Encoded dataframe.
    """
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in df:
        if df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(df[col].unique())) <= 2:
                # Train on the training data
                le.fit(df[col])
                # Transform data
                df[col] = le.transform(df[col])
            
                # Keep track of how many columns were label encoded
                le_count += 1
            
    print('%d columns were label encoded.' % le_count)
 

def split_data(X, y, test_size=0.2, random_state=42):
    """
     It will split the data into 20% for testing and rest as training.
      param: X - features of the data.
      param: y - target variable.
      param: test_size - test set size. Default is 20%.
      param: random_state - Default is 42.
      return: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

