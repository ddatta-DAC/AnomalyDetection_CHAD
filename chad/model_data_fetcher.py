import os
import pandas as pd
from pathlib import Path
import multiprocessing
from pprint import pprint
import sys

sys.path.append('./..')
sys.path.append('./../..')

try:
    from data_fetcher import data_fetcher
except:
    from .data_fetcher import data_fetcher

# -----------------------------------------------------

from pandarallel import pandarallel
pandarallel.initialize()
from joblib import delayed, Parallel
from tqdm import tqdm

from collections import OrderedDict
import numpy as np
DATA_LOC = './../generated_data_v1'

def fetch_model_data(
        data_set,
        num_neg_samples=10,
        anomaly_ratio=0.1,
        num_anom_sets=5
):
    global DATA_LOC
    df_dict, meta_data = data_fetcher.get_data(
        data_set,
        DATA_LOC = './../generated_data_v1/{}',
        one_hot = False,
        anomaly_ratio=0.5,
        num_anom_sets=5
    )
    model_data_dir = os.path.join(DATA_LOC, data_set, 'model_data')
    # Model requires 0-1 encoded data
    pos_file_path = os.path.join(model_data_dir, 'pos_samples.npy')
    neg_file_path = os.path.join(model_data_dir, 'neg_samples.npy')
    pos_x = np.load(pos_file_path)
    neg_x = np.load(neg_file_path)
    assert  pos_x.shape[0] == neg_x.shape[0]

    df_dict, meta_data = data_fetcher.get_data(
        data_set,
        one_hot=True,
        anomaly_ratio=anomaly_ratio,
        num_anom_sets=num_anom_sets
    )

    return pos_x, neg_x, df_dict
