import pandas as pd
from sklearn import preprocessing

def read_dataset(path, separator=","):
    return pd.read_csv(path, sep=separator)

def clean_and_normalize_data(data, exclude=[]):
    df_ex           = data.loc[:, data.columns.difference(exclude)]
    df_labels       = data[exclude]

    columns         = df_ex.columns
    old_indexes     = df_ex.index.values

    
    min_max_scaler  = preprocessing.MinMaxScaler()
    df_norm         = min_max_scaler.fit_transform(df_ex)
    
    df_norm = pd.DataFrame(df_norm, columns = columns, index = old_indexes)

    df_norm = pd.concat([df_norm, df_labels], axis=1)

    return df_norm

def divide_datasets(df_merged, percentage=0.67):
    
    df_divide = df_merged.sample(frac=1)
    df_train = df_divide[:int((len(df_divide))*percentage)]
    df_test = df_divide[int((len(df_divide))*percentage):]
    
    
    return df_train, df_test

def to_csv(path,dataframe):
    dataframe.to_csv(path)