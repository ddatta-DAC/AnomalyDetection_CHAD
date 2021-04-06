import sys
import os
import pandas as pd
import numpy as np

sys.path.append('./..')
sys.path.append('./../..')
sys.path.append('./../../..')
try:
    from . import utils
except:
    import utils
import argparse

import torch
from torch import FloatTensor as FT
from torch import LongTensor as LT
from torch import nn
from torch.nn import functional as F
import os
from collections import OrderedDict
import yaml
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions.normal import Normal
import math
from sklearn.cluster import MiniBatchKMeans, KMeans
import argparse

try:
    from data_fetcher import data_fetcher
except:
    from .data_fetcher import data_fetcher

try:
    from .model_dcn import DCN
except:
    from model_dcn import DCN

EPSILON = math.pow(10, -6)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Current device  >> ', DEVICE)
print('=========================== ')



def train_model(
    DATA_SET,
    data_dict,
    config
):
    global DEVICE
    
    layer_dims = config[DATA_SET]['layer_dims']
    train_df = data_dict['train']
    train_X = train_df.values
    data_dim = train_X.shape[1]

    epochs_1 = config[DATA_SET]['epochs_1']
    epochs_2 = config[DATA_SET]['epochs_2']
    K = config[DATA_SET]['k']
    batch_size = config[DATA_SET]['batch_size']
    
    dcn_obj = DCN(
        DEVICE,
        data_dim,
        layer_dims,  # Provide the half (encoder only)
        op_activation='sigmoid',
        layer_activation='sigmoid',
        dropout=0.1,
        LR=0.001,
        num_epochs_1=epochs_1,
        num_epochs_2=epochs_2,
        min_epochs=5,
        batch_size=batch_size,
        k=K,
        stop_threshold=0.05,
        checkpoint_dir=DATA_SET,
        Lambda=0.1
    )
    dcn_obj.train_model(train_X)
    print(dcn_obj.centroids)
    return dcn_obj


def test_eval(
    dcn_obj,
    data_dict,
    num_anomaly_sets
):
    test_X = data_dict['test'].values
    test_labels = [0 for _ in range(test_X.shape[0])]
    test_scores = dcn_obj.score_samples(test_X)
    auc_list = []

    for idx in range(num_anomaly_sets):
        key = 'anom_' + str(idx+1)
        anom_X = data_dict[key].values
        anom_labels = [1 for _ in range(anom_X.shape[0])]
        anom_scores = dcn_obj.score_samples(anom_X)

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

        # Vary the threshold
        thresh = _max - step
        num_anomalies = anom_X.shape[0]
        P = []
        R = [0]

        while thresh >= _min:
            sel = res_df.loc[res_df['score'] >= thresh]
            if len(sel) == 0:
                thresh -= step
                continue
            correct = sel.loc[sel['label'] == 1]
            prec = len(correct) / len(sel)
            rec = len(correct) / num_anomalies
            P.append(prec)
            R.append(rec)
            if rec >= 1.0 :
                break
            thresh -= step
            thresh = round(thresh,3)
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
    choices=['us_import1', 'us_import2', 'us_import3', 'us_import4','us_import5', 'us_import6']
)

parser.add_argument(
    '--num_runs',
    type=int,
    default=10,
    help='Number of runs'
)

args = parser.parse_args()
DATA_SET = args.DATA_SET
num_runs = args.num_runs
LOG_FILE = 'log_results_{}.txt'.format(DATA_SET)
LOGGER = utils.get_logger(LOG_FILE)
utils.log_time(LOGGER)
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
    dcn_obj = train_model(DATA_SET, data_dict, config)
    mean_aupr, std = test_eval(dcn_obj, data_dict, num_anomaly_sets)

    results.append(mean_aupr)
    LOGGER.info(' Run {}: Mean: {:4f} | Std {:4f}'.format(n,mean_aupr,std))

mean_all_runs = np.mean(results)
print('Mean AuPR over  {} runs {:4f}'.format(num_runs, mean_all_runs))
print('Details: ', results)

LOGGER.info('Mean AuPR over  {} runs {:4f} Std {:4f}'.format(num_runs, mean_all_runs, np.std(results)))
LOGGER.info(' Details ' + str(results))
utils.close_logger(LOGGER)



