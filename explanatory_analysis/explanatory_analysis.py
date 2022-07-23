import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def run(df):
    get_basic_statistics(df)
    
def get_basic_statistics(df):
    print(np.round(df.describe(), decimals=2))
    print(df.info())
    sns.heatmap(df.corr(), cmap= 'YlGnBu')

