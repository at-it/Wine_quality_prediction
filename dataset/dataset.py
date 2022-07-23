import pandas as pd
from urllib.request import urlretrieve
import os
from sklearn.model_selection import train_test_split

folder_path = os.path.join(os.path.dirname(__file__),'../')
project_dir = os.path.normpath(folder_path)

def initialize_dataset():
    url_wines_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    url_wines_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    if not file_exists(project_dir, 'dataset\winequality-white.csv'):
        filename_wines_white = project_dir + '\dataset\winequality-white.csv'
        urlretrieve(url_wines_white, filename_wines_white)
    if not file_exists(project_dir, '\dataset\winequality-red.csv'):
        filename_wines_red = project_dir + '\dataset\winequality-red.csv'
        urlretrieve(url_wines_red, filename_wines_red)

    df_wines_white = pd.read_csv(filename_wines_white, sep=';')
    df_wines_red = pd.read_csv(filename_wines_red, sep=';')
    df = pd.concat([df_wines_white, df_wines_red])
    return df

def file_exists(filename, directory):
    os.path.exists(directory + '\\' + filename)

def split_dataset_for_wine_quality_prediction(df):
    X = df.drop(['quality'],axis=1)
    y = df['quality']
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X, y, X_train, y_train, X_test, y_test
 