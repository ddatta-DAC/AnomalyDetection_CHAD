import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./..')
sys.path.append('./../..')
from tqdm import tqdm
from pathlib import Path
import multiprocessing
from pandarallel import pandarallel
pandarallel.initialize()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

# ========================================= #


def get_data(
    data_set,
    DATA_LOC = './../generated_data_v1/{}',
    one_hot=False,
    anomaly_ratio = 0.2,
    num_anom_sets = 5
):
    DATA_LOC = DATA_LOC.format(data_set)
    if not os.path.exists(DATA_LOC):
        print('ERROR :', DATA_LOC)
        exit(1)

    if one_hot is False:
        train_df = pd.read_csv(os.path.join(DATA_LOC, 'train_scaled.csv'), index_col=None)
        test_df = pd.read_csv(os.path.join(DATA_LOC, 'test_scaled.csv'), index_col=None)
        anom_2_data = pd.read_csv(os.path.join(DATA_LOC, 'anomalies_2.csv'), index_col=None)
        anom_3_data = pd.read_csv(os.path.join(DATA_LOC, 'anomalies_3.csv'), index_col=None)
        anom_size = int(anomaly_ratio * len(test_df))
        df_dict = {
            'train': train_df,
            'test': test_df,
        }

        for i in range(0,num_anom_sets):
            _df = anom_2_data.sample(n=anom_size)
            df_dict['anom_2_' + str(i)] = _df
        for i in range(0,num_anom_sets):
            _df = anom_3_data.sample(n=anom_size)
            df_dict['anom_3_' + str(i)] = _df
    else:
        train_df = pd.read_csv(os.path.join(DATA_LOC, 'train_scaled_01.csv'), index_col=None)
        test_df = pd.read_csv(os.path.join(DATA_LOC, 'test_scaled_01.csv'), index_col=None)
        anom_data_2 = pd.read_csv(os.path.join(DATA_LOC, 'anomalies_2_oneHot.csv'), index_col=None)
        anom_data_3 = pd.read_csv(os.path.join(DATA_LOC, 'anomalies_3_oneHot.csv'), index_col=None)

        anom_size = int(anomaly_ratio * len(test_df))
        df_dict = {
            'train': train_df,
            'test': test_df,
        }
        for i in range(0,num_anom_sets):
            _df = anom_data_2.sample(n=anom_size)
            df_dict['anom_2_' + str(i+1)] = _df
            
        for i in range(0,num_anom_sets):
            _df = anom_data_2.sample(n=anom_size)
            df_dict['anom_3_' + str(i+1)] = _df    
    meta_data = pd.read_csv(
        os.path.join(DATA_LOC, 'data_dimensions.csv'),
        index_col=None,
        low_memory=False
    )

    return df_dict, meta_data


# df_dict,meta_data = get_data('kddcup',True)
#
# print(len(df_dict['anom_1'].columns))
# print(len(df_dict['train'].columns))
