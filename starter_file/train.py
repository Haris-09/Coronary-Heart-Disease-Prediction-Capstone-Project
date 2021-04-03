from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.core.dataset import Dataset
from azureml.data.dataset_factory import TabularDatasetFactory

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://raw.githubusercontent.com/Haris-09/Coronary-Heart-Disease-Prediction-Capstone-Project/master/starter_file/framingham.csv"
location = "https://raw.githubusercontent.com/Haris-09/Coronary-Heart-Disease-Prediction-Capstone-Project/master/starter_file/framingham.csv"
ds = TabularDatasetFactory.from_delimited_files(path=location)



run = Run.get_context()

def clean_data(data):

    #Renaming Male column to gender : 1 - male, 0 - female
    data = data.to_pandas_dataframe().rename(columns={'male': 'gender'})
    data = data.drop(['education'],axis=1,inplace=True)
    #data = data.to_pandas_dataframe().drop('NA', axis=0)
    x_df = data.dropna(axis=0, inplace=True)
    y_df = x_df.pop("TenYearCHD")
    
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    
    args = parser.parse_args()
    
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=223)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
