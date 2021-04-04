import os
import sys
import pandas as pd
import pickle
import yaml
import re
from joblib import Parallel, delayed
import argparse
import re
import numpy as np
from pandarallel import pandarallel
pandarallel.initialize()

domain_dims = None
DIR = None

CONFIG = None
CONFIG_FILE = 'config.yaml'
ID_COL = 'PanjivaRecordID'
categorical_columns = None
use_cols = None
freq_bound = None
save_dir = None
categorical_columns = None
numeric_columns = None


# ----------------------
# Scale features
# ----------------------
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



def get_domain_dims():
    global DIR
    global save_dir
    global domain_dims
    with open(os.path.join(save_dir, 'domain_dims.pkl'),'rb') as fh:
        domain_dims = pickle.load(fh)
    return





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

def perturb_row(
    row, 
    perturb_cat_cols = 2, 
    perturb_numeric_cols = 1 
):
    global categorical_columns, numeric_columns, domain_dims
  
    row = row.copy()
    pert_cat_cols = np.random.choice( 
        categorical_columns, 
        size = perturb_cat_cols, 
        replace = False
    )
    
    for col in pert_cat_cols:
        row[col] = np.random.choice(np.arange(domain_dims[col], dtype=int ), 1)
   
    # Select a numeric column 
    
    numeric_cols = np.random.choice( numeric_columns , size = perturb_numeric_cols, replace = False)
    for nc in numeric_cols:
        val = row[nc]
        if val < 0.5:
            val += np.random.uniform(0.25,0.75) 
        else:
            val -= np.random.uniform(0.25,0.75) 
        row[nc] = val
    return row


def process():
    global categorical_columns
    test_data = pd.read_csv(os.path.join(save_dir, 'test_data_scaled.csv'))
    anomalies = test_data.parallel_apply( perturb_row, axis= 1)
    for c in categorical_columns:
        anomalies.loc[:,c] = anomalies[c].astype(int)
    anomalies_oneHot = create_oneHot_version( anomalies )
    
    anomalies.to_csv(os.path.join(save_dir, 'anomalies.csv'))
    anomalies_oneHot.to_csv(os.path.join(save_dir, 'anomalies_oneHot.csv'))
    return 



parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5' , 'us_import6'],
    default= None
)

args = parser.parse_args()
DIR = args.DIR
set_up_config(args.DIR)
process()

