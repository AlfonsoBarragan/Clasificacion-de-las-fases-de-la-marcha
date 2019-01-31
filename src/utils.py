import pandas as pd
import numpy as np

def read_dataset(path):
    return pd.read_csv(path)

def divide_datasets(df_merged):
    # Mezclamos y requetemezclamos
    df_divide = df_merged.sample(frac=1)
    # 1/3 for test  2/3 for train
    p = 0.67
    df_train = df_divide[:int((len(df_divide))*p)]
    df_test = df_divide[int((len(df_divide))*p):]
    
    return df_train,df_test

def to_csv(path,dataframe):
    np.savetxt(path, dataframe, delimiter=",")