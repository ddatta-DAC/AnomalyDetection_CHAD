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

VALID_HSCODE_LIST = []
with open('./valid_HSCodes.txt','r') as fh:
    VALID_HSCODE_LIST = fh.readlines()
VALID_HSCODE_LIST = [_.strip('\n') for _ in VALID_HSCODE_LIST]



def HSCode_filter_aux(val):
    global VALID_HSCODE_LIST
    val = str(val)
    vals = val.split(';')

    for _val in vals:
        _val = str(_val)
        _val = _val.replace('.', '')
        _val = str(_val[:8])
        if _val[:2] == '44':
            return _val
        elif _val in VALID_HSCODE_LIST:
            return _val
        else:
            continue
    return None

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
    global DIR_LOC
    global freq_bound_PERCENTILE
    global freq_bound_ABSOLUTE
    global ID_COL
    global numeric_columns
    global categorical_columns
    DATA_SOURCE = './../Data_Raw/'
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

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    use_cols = [ID_COL] + categorical_columns +  numeric_columns
    freq_bound_PERCENTILE = CONFIG['freq_bound_PERCENTILE']
    freq_bound_ABSOLUTE = CONFIG['freq_bound_ABSOLUTE']
    column_value_filters = CONFIG[DIR]['column_value_filters']

    _cols = list(use_cols)
    _cols.remove(ID_COL)
    attribute_columns = categorical_columns +  numeric_columns    
    return


def clean_Quantity(val):
    if val is None :
        return None
    if type(val) ==str and len(val) == 0:
        return None
    try:
        res = int(val.split(' ')[0]) 
    except:
        res = None
    return res

def get_regex(_type):
    global DIR
    if DIR == 'us_import1':
        if _type == 'train':
            return '.*0[1-2]_2015.csv'
        if _type == 'test':
            return '.*0[3]_2015.csv'

    if DIR == 'us_import2':
        if _type == 'train':
            return '.*0[4-5]_2015.csv'
        if _type == 'test':
            return '.*0[6]_2015.csv'

    if DIR == 'us_import3':
        if _type == 'train':
            return '.*0[2-3]_2016.csv'
        if _type == 'test':
            return '.*0[4]_2016.csv'
        
    if DIR == 'us_import4':
        if _type == 'train':
            return '.*0[4-5]_2016.csv'
        if _type == 'test':
            return '.*0[6]_2016.csv'
    
    if DIR == 'us_import5':
        if _type == 'train':
            return '.*0[1-2]_2017.csv'
        if _type == 'test':
            return '.*0[3]_2017.csv'
    
    if DIR == 'us_import6':
        if _type == 'train':
            return '.*0[5-6]_2015.csv'
        if _type == 'test':
            return '.*0[7]_2015.csv'
    return '*.csv'


def get_files(DIR, _type='all'):
    global DATA_SOURCE
    data_dir = DATA_SOURCE

    regex = get_regex(_type)
    print(regex)
    c = glob.glob(os.path.join(data_dir, '*'))

    def glob_re(pattern, strings):
        return filter(re.compile(pattern).match, strings)

    files = sorted([_ for _ in glob_re(regex, c)])

    print('DIR ::', DIR, ' Type ::', _type, 'Files count::', len(files))
    return files

def remove_low_frequency_values(df):
    global id_col
    global freq_bound_PERCENTILE
    global freq_bound_ABSOLUTE
    global categorical_columns

    freq_column_value_filters = {}
    feature_cols = list(categorical_columns)
    print('feature columns ::', feature_cols)
    # ----
    # Figure out which entities are to be removed
    # ----

    counter_df = pd.DataFrame(columns=['domain', 'count'])
    for c in feature_cols:
        count = len(set(df[c]))
        counter_df = counter_df.append({
            'domain': c, 'count': count
        }, ignore_index=True)

        z = np.percentile(
            list(Counter(df[c]).values()), 5)
       

    counter_df = counter_df.sort_values(by=['count'], ascending=True)
    print(' Data frame of Number of values', counter_df)

    for c in list(counter_df['domain']):

        values = list(df[c])
        freq_column_value_filters[c] = []
        obj_counter = Counter(values)
        for _item, _count in obj_counter.items():
            if _count < freq_bound_PERCENTILE or _count < freq_bound_ABSOLUTE:
                freq_column_value_filters[c].append(_item)

    print('Removing :: ')
    for c, _items in freq_column_value_filters.items():
        print('column : ', c, 'count', len(_items))

    print(' DF length : ', len(df))
    for col, val in freq_column_value_filters.items():
        df = df.loc[~df[col].isin(val)]

    print(' DF length : ', len(df))
    return df


def apply_value_filters(list_df):
    global column_value_filters

    if type(column_value_filters) != bool:
        list_processed_df = []
        for df in list_df:
            for col, val in column_value_filters.items():
                df = df.loc[~df[col].isin(val)]
            list_processed_df.append(df)
        return list_processed_df
    return list_df



def attribute_cleanup(list_df):
    new_list = []
    for _df in list_df:
        _df['HSCode'] = _df['HSCode'].parallel_apply(HSCode_filter_aux)
        _df = _df.dropna(subset=['HSCode'])
        _df['Quantity'] = _df['Quantity'].parallel_apply(clean_Quantity) 
        _df = _df.dropna(subset=['Quantity'])
        print(' In HSCode clean up , length of dataframe ', len(_df))
        new_list.append(_df)
    return new_list

def clean_train_data():
    global DIR
    global CONFIG
    global DIR_LOC
    global categorical_columns
    global numeric_columns
    
    files = get_files(DIR, 'train')
    print('Columns read ', use_cols)
    list_df = [
        pd.read_csv(_file, usecols=use_cols, low_memory=False) for _file in files
    ]
    list_df = [_.dropna() for _ in list_df]
    list_df = attribute_cleanup(list_df)
    list_df_1 = apply_value_filters(list_df)
    master_df = None

    for df in list_df_1:
        if master_df is None:
            master_df = pd.DataFrame(df, copy=True)
        else:
            master_df = master_df.append(
                df,
                ignore_index=True
            )
    master_df = remove_low_frequency_values(master_df)
    print(len(master_df))
    return master_df



def convert_to_ids(
        df,
        save_dir
):
    global id_col
    global freq_bound
    global categorical_columns, numeric_columns
    pandarallel.initialize()

    feature_columns = categorical_columns
    dict_DomainDims = {}
    col_val2id_dict = {}

    for col in feature_columns:
        vals = list(set(df[col]))
        vals = list(sorted(vals))

        id2val_dict = {
            e[0]: e[1]
            for e in enumerate(vals, 0)
        }
        print(' > ', col, ':', len(id2val_dict))

        val2id_dict = {
            v: k for k, v in id2val_dict.items()
        }
        col_val2id_dict[col] = val2id_dict

        # Replace
        df[col] = df.parallel_apply(
            replace_attr_with_id,
            axis=1,
            args=(col, val2id_dict,)
        )

        dict_DomainDims[col] = len(id2val_dict)

    print(' dict_DomainDims ', dict_DomainDims)

    # -------------
    # Save the domain dimensions
    # -------------

    file = 'domain_dims.pkl'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    f_path = os.path.join(save_dir, file)

    with open(f_path, 'wb') as fh:
        pickle.dump(
            dict_DomainDims,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    file = 'col_val2id_dict.pkl'
    f_path = os.path.join(save_dir, file)

    with open(f_path, 'wb') as fh:
        pickle.dump(
            col_val2id_dict,
            fh,
            pickle.HIGHEST_PROTOCOL
        )
    return df, col_val2id_dict

def replace_attr_with_id(row, attr, val2id_dict):
    val = row[attr]
    if val not in val2id_dict.keys():
        return None
    else:
        return val2id_dict[val]

def order_cols(df):
    global categorical_columns
    global numeric_columns
    global ID_COL

    ord_cols = [ID_COL] + categorical_columns + numeric_columns
    return df[ord_cols]

def setup_testing_data(
        test_df,
        train_df,
        col_val2id_dict
):
    global id_col
    global save_dir
    global categorical_columns, numeric_columns
    test_df = test_df.dropna()

    # Replace with None if ids are not in train_set
    feature_cols = list(categorical_columns)

    for col in feature_cols:
        valid_items = list(col_val2id_dict[col].keys())
        test_df = test_df.loc[test_df[col].isin(valid_items)]

    # First convert to to ids
    for col in feature_cols:
        val2id_dict = col_val2id_dict[col]
        test_df[col] = test_df.parallel_apply(
            replace_attr_with_id,
            axis=1,
            args=(
                col,
                val2id_dict,
            )
        )
    test_df = test_df.dropna()
    test_df = test_df.drop_duplicates(subset= categorical_columns + numeric_columns)
    
    test_df = order_cols(test_df)

    print(' Length of testing data', len(test_df))
    test_df = order_cols(test_df)
    return test_df

def create_train_test_sets():
    global use_cols
    global DIR
    global save_dir
    global column_value_filters
    global CONFIG
    global DIR_LOC
    global categorical_columns,numeric_columns
    
    train_df_file = os.path.join(save_dir, 'train_data.csv')
    test_df_file = os.path.join(save_dir, 'test_data.csv')
    column_valuesId_dict_file = 'column_valuesId_dict.pkl'
    column_valuesId_dict_path = os.path.join(
        save_dir,
        column_valuesId_dict_file
    )

    # --- Later on - remove using the saved file ---- #

    if os.path.exists(train_df_file) and os.path.exists(test_df_file) :
        train_df = pd.read_csv(train_df_file)
        test_df = pd.read_csv(test_df_file)
        with open(column_valuesId_dict_path, 'rb') as fh:
            col_val2id_dict = pickle.load(fh)

#         return train_df, test_df, col_val2id_dict

    train_df = clean_train_data()
    train_df = order_cols(train_df)
    train_df, col_val2id_dict = convert_to_ids(
        train_df,
        save_dir
    )
    
    train_df = train_df.drop_duplicates(subset=categorical_columns + numeric_columns)
   
    print('Length of train data ', len(train_df))
    train_df = order_cols(train_df)

    '''
         test data preprocessing
    '''
    # combine test data into 1 file :
    test_files = get_files(DIR, 'test')
    list_test_df = [
        pd.read_csv(_file, low_memory=False, usecols=use_cols)
        for _file in test_files
    ]
    list_test_df = [_.dropna() for _ in list_test_df]
    list_test_df = attribute_cleanup(list_test_df)

    test_df = None
    for _df in list_test_df:
        if test_df is None:
            test_df = _df
        else:
            test_df = test_df.append(_df)

    print('size of  Test set ', len(test_df))
    test_df = setup_testing_data(
        test_df,
        train_df,
        col_val2id_dict
    )

    test_df.to_csv(test_df_file, index=False)
    train_df.to_csv(train_df_file, index=False)

    # Save data_dimensions.csv ('column', dimension')
    dim_df = pd.DataFrame(columns=['column', 'dimension'])
    for col in categorical_columns:
        _count = len(col_val2id_dict[col])
        dim_df = dim_df.append({'column': col, 'dimension': _count}, ignore_index=True)

    dim_df.to_csv(os.path.join(save_dir, 'data_dimensions.csv'), index=False)

    # -----------------------
    # Save col_val2id_dict
    # -----------------------
    with open(column_valuesId_dict_path, 'wb') as fh:
        pickle.dump(col_val2id_dict, fh, pickle.HIGHEST_PROTOCOL)

    return train_df, test_df, col_val2id_dict




parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5' , 'us_import6'],
    default= None
)

args = parser.parse_args()
DIR = args.DIR
# -------------------------------- #
set_up_config(args.DIR)
create_train_test_sets()

