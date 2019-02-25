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

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    # Code from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()