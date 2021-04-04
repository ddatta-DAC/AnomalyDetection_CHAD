import pandas as pd
import os
import sys
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./../..')
sys.path.append('./..')
import glob
from tqdm import tqdm
import joblib
import multiprocessing as mp
from joblib import Parallel, delayed
import argparse
import re
import yaml
from collections import Counter
import pickle
from pandarallel import pandarallel
pandarallel.initialize()
from sklearn.preprocessing import MinMaxScaler

CONFIG = None
DIR_LOC = None
CONFIG = None
CONFIG_FILE = 'config.yaml'
ID_COL = 'PanjivaRecordID'
categorical_columns = None
use_cols = None
freq_bound = None
save_dir = None
categorical_columns = None
numeric_columns = None



def get_domain_dims():
    global DIR
    global save_dir
    global domain_dims
    with open(os.path.join(save_dir, 'domain_dims.pkl'),'rb') as fh:
        domain_dims = pickle.load(fh)
    return


def set_up_config(_DIR=None):
    global DIR
    global CONFIG
    global CONFIG_FILE
    global use_cols
    global num_neg_samples_ape
    global save_dir
    global column_value_filters
    global num_neg_samples
    global DATA_SOURCE 
    global ID_COL
    global numeric_columns
    global categorical_columns, numeric_columns
    
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is not None:
        DIR = _DIR
        CONFIG['DIR'] = _DIR
    else:
        DIR = CONFIG['DIR']
    numeric_columns = list(sorted(CONFIG['numeric_columns']))
    categorical_columns = list(sorted(CONFIG['categorical_columns']))
    ID_COL = 'PanjivaRecordID'
    DIR_LOC = re.sub('[0-9]', '', DIR)
#     DATA_SOURCE = os.path.join(DATA_SOURCE, DIR_LOC)
    save_dir = CONFIG['save_dir']
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(
        CONFIG['save_dir'],
        DIR
    )
    DATA_SOURCE = save_dir
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    use_cols = [ID_COL] + categorical_columns +  numeric_columns
    freq_bound_PERCENTILE = CONFIG['freq_bound_PERCENTILE']
    freq_bound_ABSOLUTE = CONFIG['freq_bound_ABSOLUTE']
    column_value_filters = CONFIG[DIR]['column_value_filters']

    _cols = list(use_cols)
    _cols.remove(ID_COL)
    attribute_columns = categorical_columns +  numeric_columns    
    get_domain_dims()
    
    return




def feature_transform(DIR):
    global categorical_columns, numeric_columns
    global DATA_SOURCE
    
    train_data = pd.read_csv(os.path.join(DATA_SOURCE, 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(DATA_SOURCE, 'test_data.csv'))
    
    for num in numeric_columns:
        scaler_obj = MinMaxScaler()
        scaler_obj.fit( train_data[num].values.reshape(-1,1))
        train_data.loc[:,num] = scaler_obj.transform(train_data[num].values.reshape(-1,1)).reshape(-1)
        test_data.loc[:,num] = scaler_obj.transform(test_data[num].values.reshape(-1,1)).reshape(-1)
    return train_data, test_data




# -------------------------------
#  generate one hot version 
# -------------------------------
def create_oneHot_version( df ):
    global categorical_columns, numeric_columns, ID_COL, domain_dims
    df1 = df.copy(deep=True) 
    
    for _c in categorical_columns:
        df1[_c]  = pd.Categorical(df1[_c], categories= np.arange(domain_dims[_c],dtype=int ))
        df1 = pd.get_dummies(df1)
    cols = [ _ for _ in df1.columns if _ not in numeric_columns and _!=ID_COL]
    ordered_cols = [ID_COL] + cols + numeric_columns
    df1 = df1[ordered_cols]    
    return df1


def process():
    train_data_scaled, test_data_scaled =  feature_transform(DIR)
    train_scaled_01 = create_oneHot_version( train_data_scaled )
    test_scaled_01 = create_oneHot_version( test_data_scaled )

    # ------------------
    # Save data
    # ------------------

    train_data_scaled.to_csv(os.path.join(save_dir, 'train_scaled.csv'), index=None)
    test_data_scaled.to_csv(os.path.join(save_dir, 'test_scaled.csv'), index=None)
    train_scaled_01.to_csv(os.path.join(save_dir, 'train_scaled_01.csv'), index=None)
    test_scaled_01.to_csv(os.path.join(save_dir, 'test_scaled_01.csv'), index=None)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5' , 'us_import6'],
    default= None
)


args = parser.parse_args()
DIR = args.DIR
# -------------------------------- #
set_up_config(args.DIR)
process()