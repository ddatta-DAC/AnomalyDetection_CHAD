import sys
import os
import pandas as pd
import numpy as np

sys.path.append('./..')
sys.path.append('./../..')
import torch
import math
import yaml
from sklearn.metrics import auc
from tqdm import tqdm
from collections import OrderedDict
from matplotlib import pyplot as plt
from pathlib import Path
import argparse
import multiprocessing
from pprint import pprint
from time import time
from datetime import datetime
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print('Current device  >> ', DEVICE)
# ===============================================
try:
    from data_fetcher import data_fetcher
except:
    from .data_fetcher import data_fetcher
try:
    from model import model_FAER_container as Model
except:
    from .model import model_FAER_container as Model
try:
    from utils import create_config
except:
    from .utils import create_config
try:
    import utils
except:
    from . import utils

ID_COL = 'PanjivaRecordID'

def execute_run(DATA_SET):
    global ID_COL
    global LOGGER
    encoder_structure_config, decoder_structure_config, loss_structure_config, latent_dim = create_config(DATA_SET)
    config_file = 'architecture_config.yaml'

    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)

    anomaly_ratio = config['anomaly_ratio']
    LR = config['LR']
    batch_size =  config['batch_size']
    epochs = config['epochs']
    dropout = config['ae_dropout']

    ae_model = Model(
        DEVICE,
        latent_dim,
        encoder_structure_config,
        decoder_structure_config,
        loss_structure_config,
        optimizer='Adam',
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=LR,
        dropout=dropout
    )

    print(ae_model.network_module)
    num_anomaly_sets = 5
    data_dict, _ = data_fetcher.get_data(
        DATA_SET,
        one_hot=True,
        num_anom_sets= num_anomaly_sets,
        anomaly_ratio= anomaly_ratio
    )
    train_df = data_dict['train']
    try:
        del train_df[ID_COL]
    except:
        pass

    train_X = train_df.values


    ae_model.train_model(
        train_X
    )

    test_df = data_dict['test']
    try:
        del test_df[ID_COL]
    except:
        pass
    test_norm_X = test_df.values



    auc_result = {}
    ae_model.mode = 'test'

    def _normalize_(val, _min, _max):
        return (val - _min) / (_max - _min)

    for anomaly_key in ['anom_2_', 'anom_3_']:
        auc_list = []
        for idx in range(num_anomaly_sets):
            key = anomaly_key + str(idx + 1)
            anom_df = data_dict[key]
            try:
                del anom_df[ID_COL]
            except:
                pass

            test_anom_X = anom_df.values

            x1 = test_norm_X
            x2 = test_anom_X

            x1_scores = ae_model.get_score(x1)
            x2_scores = ae_model.get_score(x2)

            res_data = []
            labels = [1 for _ in range(x1.shape[0])] + [0 for _ in range(x2.shape[0])]
            _scores = np.concatenate([x1_scores, x2_scores], axis=0)
            for i, j in zip(_scores, labels):
                res_data.append((i, j))

            res_df = pd.DataFrame(res_data, columns=['score', 'label'])
            res_df = res_df.sort_values(by=['score'], ascending=False)

            _max = max(res_df['score'])
            _min = min(res_df['score'])

            res_df['score'] = res_df['score'].parallel_apply(
                _normalize_,
                args=(_min, _max,)
            )

            _max = max(res_df['score'])
            _min = min(res_df['score'])

            step = (_max - _min) / 100

            # Vary the threshold
            thresh = _max - step
            thresh = round(thresh, 3)
            num_anomalies = x2.shape[0]
            P = []
            R = [0]
            while thresh >= _min :
                sel = res_df.loc[res_df['score'] >= thresh]
                if len(sel) == 0:
                    thresh -= step
                    continue

                correct = sel.loc[sel['label'] == 0]

                prec = len(correct) / len(sel)
                rec = len(correct) / num_anomalies
                P.append(prec)
                R.append(rec)
                thresh -= step
                thresh = round(thresh, 3)

            P = [P[0]] + P
            pr_auc = auc(R, P)

            print("AUC : {:0.4f} ".format(pr_auc))
            auc_list.append(pr_auc)

        mean_auc = np.mean(auc_list)
        print(' (Mean) AUC {:0.4f} '.format(mean_auc))
        auc_result[anomaly_key] = mean_auc
    return auc_result


# ==========================================================

parser = argparse.ArgumentParser(description='Run the model ')
parser.add_argument(
    '--DATA_SET',
    type=str,
    help=' Which data set ?',
    default=None,
    choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5', 'us_import6']
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
LOGGER = utils.get_logger(LOG_FILE)
utils.log_time(LOGGER)
LOGGER.info(DATA_SET)

results = {}
for n in range(1, num_runs + 1):
    auc_result = execute_run(DATA_SET)
    for key,_aupr in auc_result.items():
        if key not in results.keys():
            results[key] = []
        results[key].append(_aupr)
        LOGGER.info("Run {}:  Anomaly type {} AuPR: {:4f}".format(n, key, _aupr))
#--------------------
for key, _aupr in results.items():
    mean_all_runs = np.mean(_aupr)
    log_op = 'Mean AuPR over runs {} | {} | {:5f} Std {:.5f}'.format(num_runs, key, mean_all_runs, np.std(_aupr))
    LOGGER.info(log_op)
    print(log_op)
    LOGGER.info(' Details ' + str(_aupr))

utils.close_logger(LOGGER)
