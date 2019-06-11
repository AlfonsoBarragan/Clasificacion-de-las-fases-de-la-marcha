import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
import pickle
from os import scandir, getcwd
import random

################### Datasets section #####################
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

def normalize_data(dataframe):
    min_max_scaler  = preprocessing.MinMaxScaler()
    norm_values         = min_max_scaler.fit_transform(dataframe)
    
    dataframe_norm  = pd.DataFrame(data=norm_values, columns=dataframe.columns) 
     
    return dataframe_norm

def divide_datasets(df_merged, percentage=0.67):
    
    df_divide = df_merged.sample(frac=1)
    df_train = df_divide[:int((len(df_divide))*percentage)]
    df_test = df_divide[int((len(df_divide))*percentage):]    
    
    return df_train, df_test

def divide_files(list_of_files, percentage=0.67):

    list_train  = []
    list_test   = []
    
    for i in range(len(list_of_files)):
        i_list = random.randint(1, 100)

        if i_list <= percentage * 100:
            list_train.append(list_of_files[i])
        else:
            list_test.append(list_of_files[i])

    return list_train, list_test

def to_csv(path,dataframe):
    dataframe.to_csv(path)


def insert_row_in_pos(pos, row_value, df):
	# Funciona con objetos de tipo Series de pandas.

    data_half_low, data_half_big = df[:pos], df[pos:]
    data_half_low = data_half_low.append(row_value, ignore_index = True)
    data_half_low = data_half_low.append(data_half_big, ignore_index = True)
	
    return data_half_low


################## Extras section ###########################
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

def binarySearch(alist, item):
    # Code from https://stackoverflow.com/questions/34420006/binary-search-python-3-5
    first = 0
    last = len(alist)-1
    found = False
    
    while first<=last and not found:
        midpoint = (first + last)//2
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1

    return found, midpoint

def ls(ruta = getcwd()):
    # Code from https://es.stackoverflow.com/questions/24278/
    return [arch.name for arch in scandir(ruta) if arch.is_file()]

def write_list_to_file(list_to_write, output_file):
    file = open(output_file, 'w')
        
    for element in list_to_write:
        file.write("'{}',".format(element))

    file.close()

####################### Model section ############################

def save_model(model, test):
    
    features    = test.columns[:32]
    Xtest       = test[features]
    Ytest       = test['groups']
    
    # Save to file in the current working directory
    pkl_filename = "clasificador_swing_ground_move.pkl"  
    with open(pkl_filename, 'wb') as file:  
        pickle.dump(model, file)
    
    # Load from file
    with open(pkl_filename, 'rb') as file:  
        pickle_model = pickle.load(file)
    
    # Calculate the accuracy score and predict target values
    score = pickle_model.score(Xtest, Ytest)  
    print("Test score: {0:.2f} %".format(100 * score))  
    
def load_model(joblib_file, test, acc_opt=-1):
    
    if acc_opt == 0:
        features    = test.columns[:32]
        Xtest       = test[features]
        Ytest       = test['groups']
            
        # Load from file
        joblib_model = joblib.load(joblib_file)
        
        # Calculate the accuracy and predictions
        score = joblib_model.score(Xtest, Ytest)  
        print("Test score: {0:.2f} %".format(100 * score))
        
        return joblib_model

    else:
        joblib_model = joblib.load(joblib_file)
        return joblib_model
