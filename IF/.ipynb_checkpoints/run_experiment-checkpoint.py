#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyod
from pyod.models.lscp import LSCP
import pandas as pd
import numpy as np
import sys
import os
sys.path.append('./..')
sys.path.append('./../..')
import pandas as pd
import yaml
from torch import FloatTensor as FT
import numpy as np
from pyod.models.lof import LOF
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torch
from pprint import pprint
from collections import OrderedDict
try:
    from data_fetcher import data_fetcher
except:
    from .data_fetcher import data_fetcher
import argparse
from pathlib import Path
import multiprocessing
from pprint import pprint
import torch
import math
import yaml
import matplotlib.pyplot  as plt
from sklearn.metrics import auc
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.loci import LOCI
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
import logging
import logging.handlers
from time import time
from datetime import datetime
from sklearn.svm import OneClassSVM as OCSVM

def get_logger():
    global LOG_FILE

    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    OP_DIR = os.path.join('Logs')

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    handler = logging.FileHandler(os.path.join(OP_DIR, LOG_FILE))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('Log start :: ' + str(datetime.now()))
    return logger


def log_time(logger):
    logger.info(str(datetime.now()) + '| Time stamp ' + str(time()))


def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    logging.shutdown()
    return


def train_model(DATA_SET, data_dict, config):
    model_obj = OCSVM(
        kernel='rbf',
        gamma='auto',
        nu=0.1,
        shrinking=True,
        cache_size=1000,
        verbose=True,
        max_iter=-1
    )
    train_df = data_dict['train']
    train_X = train_df.values
    model_obj.fit(train_X)
    return model_obj

def test_eval(model_obj, data_dict, num_anomaly_sets):
    test_X = data_dict['test'].values
    test_labels = [0 for _ in range(test_X.shape[0])]

    test_scores = model_obj.score_samples(test_X)
    auc_list = []

    for idx in range(num_anomaly_sets):
        key = 'anom_' + str(idx + 1)
        anom_X = data_dict[key].values
        anom_labels = [1 for _ in range(anom_X.shape[0])]
        anom_scores = model_obj.score_samples(anom_X)

        combined_scores = np.concatenate([anom_scores, test_scores], axis=0)
        combined_labels = np.concatenate([anom_labels, test_labels], axis=0)

        res_data = []
        for i, j in zip(combined_scores, combined_labels):
            res_data.append((i, j))
        res_df = pd.DataFrame(res_data, columns=['score', 'label'])

        #  Normalize values
        def _normalize_(val, _min, _max):
            return (val - _min) / (_max - _min)

        _max = max(combined_scores)
        _min = min(combined_scores)

        res_df['score'] = res_df['score'].parallel_apply(
            _normalize_,
            args=(_min, _max,)
        )

        res_df = res_df.sort_values(by=['score'], ascending=False)
        _max = max(res_df['score'])
        _min = min(res_df['score'])
        step = (_max - _min) / 100

        # OCSVM score samples score anomalies lower
        # Vary the threshold
        thresh = _min + step
        num_anomalies = anom_X.shape[0]
        P = []
        R = [0]

        while thresh <= _max:
            sel = res_df.loc[res_df['score'] <= thresh]
            if len(sel) == 0:
                thresh -= step
                continue
            correct = sel.loc[sel['label'] == 1]
            prec = len(correct) / len(sel)
            rec = len(correct) / num_anomalies
            P.append(prec)
            R.append(rec)
            if rec >= 1.0:
                break
            thresh += step
            thresh = round(thresh, 3)
        P = [P[0]] + P
        from sklearn.metrics import auc

        pr_auc = auc(R, P)
        auc_list.append(pr_auc)

        print("AUC : {:0.4f} ".format(pr_auc))
        try:
            plt.figure()
            plt.title('PR Curve' + str(pr_auc))
            plt.plot(R, P)
            plt.show()
        except:
            pass

    _mean = np.mean(auc_list)
    _std = np.std(auc_list)
    print(' Mean AUC ', np.mean(auc_list))
    print(' AUC std', np.std(auc_list))
    return _mean, _std

# ==============================================================

parser = argparse.ArgumentParser(description='Run the model ')
parser.add_argument(
    '--DATA_SET',
    type=str,
    help=' Which data set ?',
    default=None,
    choices=['kddcup', 'kddcup_neptune', 'nsl_kdd', 'nb15','gureKDD']
)

parser.add_argument(
    '--num_runs',
    type=int,
    default=5,
    help='Number of runs'
)

args = parser.parse_args()
DATA_SET = args.DATA_SET
num_runs = args.num_runs
LOG_FILE = 'log_results_{}.txt'.format(DATA_SET)
LOGGER = get_logger()
log_time(LOGGER)
LOGGER.info(DATA_SET)
config_file = 'config.yaml'
with open(config_file, 'r') as fh:
    config = yaml.safe_load(fh)

num_anomaly_sets = config[DATA_SET]['num_anomaly_sets']
anomaly_ratio = config[DATA_SET]['anomaly_ratio']
results = []

for n in range(1,num_runs+1):
    data_dict, _ = data_fetcher.get_data(
        DATA_SET,
        one_hot=True,
        num_anom_sets=num_anomaly_sets,
        anomaly_ratio=anomaly_ratio
    )

    model_obj = train_model(DATA_SET, data_dict, config)
    mean_aupr, std = test_eval(model_obj, data_dict, num_anomaly_sets)
    results.append(mean_aupr)
    LOGGER.info(' Run {}: Mean: {:4f} | Std {:4f}'.format(n,mean_aupr,std))

mean_all_runs = np.mean(results)
print('Mean AuPR over  {} runs {:4f}'.format(num_runs, mean_all_runs))
print('Details: ', results)

LOGGER.info('Mean AuPR over  {} runs {:4f} Std {:4f}'.format(num_runs, mean_all_runs, np.std(results)))
LOGGER.info(' Details ' + str(results))
close_logger(LOGGER)

