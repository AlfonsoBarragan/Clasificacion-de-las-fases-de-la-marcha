import pandas as pd
import numpy as np

def read_dataset(path, separator=","):
    return pd.read_csv(path, sep=separator)

def divide_datasets(df_merged, percentage=0.67):
    df_divide = df_merged.sample(frac=1)
    
    df_train = df_divide[:int((len(df_divide))*percentage)]
    df_test = df_divide[int((len(df_divide))*percentage):]
    
    return df_train,df_test

def to_csv(path,dataframe):
    np.savetxt(path, dataframe, delimiter=",", fmt='%s')