import torch
import random
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import auc
sys.path.append('./..')
sys.path.append('./../..')
try:
    from .networks.AE import FC_dec
    from .networks.AE import FC_enc
except:
    from networks.AE import FC_dec
    from networks.AE import FC_enc

from torch import FloatTensor as FT
from torch import LongTensor as LT
from torch import nn
from torch.nn import functional as F
import os
from collections import OrderedDict
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from sklearn.cluster import MiniBatchKMeans, KMeans
import argparse
try:
    from . import utils
except:
    import utils
try:
    from data_fetcher import data_fetcher
except:
    from data_fetcher import data_fetcher

try:
    from deepSVDD import DeepSVDD
except:
    from .deepSVDD import DeepSVDD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device ::', DEVICE)
ID_COL = 'PanjivaRecordID'
################################################################################
# Settings
################################################################################

def main(
        data_dict,
        layer_dims,
        objective='soft-boundary',
        config=None,
        NU = None
):
    global DEVICE
    global ID_COL
    
    LR = config['LR']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    warm_up_epochs = config['warm_up_epochs']
    ae_epochs = config['ae_epochs']
    num_anomaly_sets = config['num_anomaly_sets']
    
    
    train_df = data_dict['train']
    try:
        del train_df[ID_COL]
    except:
        pass
    
    train_X = train_df.values
    fc_layer_dims = [train_X.shape[1]] + list(layer_dims)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(
        DEVICE,
        objective=objective,
        nu = NU
    )
    deep_SVDD.set_network(fc_layer_dims)

    # Train model on dataset
    deep_SVDD.train(
        train_X,
        LR = LR,
        num_epochs = num_epochs,
        batch_size= batch_size,
        ae_epochs = ae_epochs,
        warm_up_epochs=warm_up_epochs
    )

    # Test model

    test_df = data_dict['test']
    try:
        del test_df[ID_COL]
    except:
        pass
    test_X = test_df.values
    # =======================================================
    
    test_scores = deep_SVDD.test(test_X)
    test_labels = [0 for _ in range(test_X.shape[0])]
    auc_list = []

    for idx in range(num_anomaly_sets):
        key = 'anom_' + str(idx + 1)
        anom_df = data_dict[key]
        try:
            del anom_df[ID_COL]
        except:
            pass
        
        anom_X = anom_df.values
        anom_labels = [1 for _ in range(anom_X.shape[0])]
        anom_scores = deep_SVDD.test(anom_X)

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
            if rec >= 1.0:
                break
            thresh -= step
            thresh = round(thresh, 3)
        P = [P[0]] + P

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

# ==================================================

parser = argparse.ArgumentParser(description='Run the model ')
parser.add_argument(
    '--DIR',
    type=str,
    help=' Which data set ?',
    default='us_import1',
    choices=['us_import1', 'us_import2', 'us_import3', 'us_import4','us_import5', 'us_import6']
)
parser.add_argument(
    '--nu',
    type=float,
    default=0.05
)

parser.add_argument(
    '--num_runs',
    type=int,
    default=1,
    help='Number of runs'
)

args = parser.parse_args()
DATA_SET = args.DIR
num_runs = args.num_runs
nu = args.nu
LOG_FILE = 'log_results_{}.txt'.format(DATA_SET)
LOGGER = utils.get_logger(LOG_FILE)
utils.log_time(LOGGER)
LOGGER.info(DATA_SET)
config_file = 'config.yaml'

with open(config_file, 'r') as fh:
    config = yaml.safe_load(fh)

num_anomaly_sets = config['num_anomaly_sets']
anomaly_ratio = config['anomaly_ratio']
config = config
print(config)
layer_dims = config['layer_dims']

objectives =['soft-boundary','one-class']
LOGGER.info(str(config))

data_dict, _ = data_fetcher.get_data(
    DATA_SET,
    one_hot=True,
    num_anom_sets=num_anomaly_sets,
    anomaly_ratio=anomaly_ratio
)

aupr_list = [] 
for n in range(1, num_runs + 1):
    aupr, std = main(
        data_dict,
        layer_dims,
        objective='soft-boundary',
        config=config,
        NU=nu
    )
    aupr_list.append(aupr)
    LOGGER.info('soft-boundary || Run {} : AuPR: {:4f} '.format(n, aupr))

LOGGER.info('AuPR  Objective {} Mean {:.4f}  Std {:.4f}'.format( 'soft-boundary', np.mean(aupr_list),  np.std(aupr_list)))


aupr_list = [] 
for n in range(1, num_runs + 1):
    aupr, std = main(
        data_dict,
        layer_dims,
        objective='one-class',
        config=config,
        NU=nu
    )
    aupr_list.append(aupr)
    LOGGER.info(' one-class || Run {} : AuPR: {:4f} '.format(n, aupr))

LOGGER.info('AuPR  Objective {} Mean {:.4f}  Std {:.4f}'.format( 'one-class', np.mean(aupr_list),  np.std(aupr_list)))
utils.close_logger(LOGGER)



