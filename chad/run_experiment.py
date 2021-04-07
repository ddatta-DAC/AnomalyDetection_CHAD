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

ID_COL = 'PanjivaRecordID'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print('Current device  >> ', DEVICE)
# ===============================================
try:
    from data_fetcher import data_fetcher
except:
    from .data_fetcher import data_fetcher
try:
    from model import model_container as Model
except:
    from .model import model_container as Model

try:
    from . import model_data_fetcher
except:
    import model_data_fetcher

try:
    from . import utils
except:
    import utils

def _normalize_(val, _min, _max):
    return (val - _min) / (_max - _min)
def execute_run(DATA_SET):
    global LOGGER
    global ID_COL
    encoder_structure_config, decoder_structure_config, loss_structure_config, latent_dim = utils.create_config(DATA_SET)
    ae_model = None

    config_file = 'architecture_config.yaml'

    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)
    
    burn_in_epochs = config['burn_in_epochs']
    phase_2_epochs = config['phase_2_epochs']
    phase_3_epochs = config['phase_3_epochs']
    batch_size = config['batch_size']
    ae_dropout = config['ae_dropout']
    fc_dropout = config['fc_dropout']
    anomaly_ratio = config['anomaly_ratio']
    LR = config['LR']
    max_gamma = config['max_gamma']

    num_anomaly_sets = 5
    pos, neg, data_dict = model_data_fetcher.fetch_model_data(
        DATA_SET,
        num_anom_sets=num_anomaly_sets,
        anomaly_ratio=anomaly_ratio
    )

    not_converged = True

    while not_converged:
        ae_model = Model(
            DATA_SET,
            DEVICE,
            latent_dim,
            encoder_structure_config,
            decoder_structure_config,
            loss_structure_config,
            batch_size=batch_size,
            fc_dropout=fc_dropout,
            ae_dropout=ae_dropout,
            learning_rate=LR,
            max_gamma=max_gamma,
            burn_in_epochs=burn_in_epochs,
            phase_2_epochs=phase_2_epochs,
            phase_3_epochs=phase_3_epochs
        )

        print(ae_model.network_module)

        _, epoch_losses_phase_3 = ae_model.train_model(
            pos,
            neg
        )
        print(epoch_losses_phase_3)
        if epoch_losses_phase_3[-1] < epoch_losses_phase_3[0]:
            not_converged = False

        if epoch_losses_phase_3[-1] >= 0.1:
            not_converged = True

    test_norm_df = data_dict['test']
    del test_norm_df[ID_COL]
    test_norm_X = test_norm_df.values

    ae_model.mode = 'test'
    auc_result = {}
    for anomaly_key in ['anom_2_', 'anom_3_']:
        auc_list = []
        for idx in range(1, num_anomaly_sets + 1):
            key = anomaly_key + str(idx)
            test_anom_df = data_dict[key]
            del test_anom_df[ID_COL]
            test_anom_X = test_anom_df.values

            x1 = test_norm_X
            x2 = test_anom_X
            x1_scores = ae_model.get_score(x1)
            x2_scores = ae_model.get_score(x2)

            res_data = []
            labels = [1 for _ in range(x1.shape[0])] + [0 for _ in range(x2.shape[0])]
            _scores = np.concatenate([x1_scores, x2_scores], axis=0)

            for i, j in zip(_scores, labels):
                res_data.append((i[0], j))

            res_df = pd.DataFrame(res_data, columns=['score', 'label'])
            res_df = res_df.sort_values(by=['score'], ascending=True)

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
            thresh = _min + step
            thresh = round(thresh, 3)
            num_anomalies = x2.shape[0]
            print('Num anomalies', num_anomalies)
            P = []
            R = [0]

            while thresh <= _max + step:
                sel = res_df.loc[res_df['score'] <= thresh]
                if len(sel) == 0:
                    thresh += step
                    continue

                correct = sel.loc[sel['label'] == 0]
                prec = len(correct) / len(sel)
                rec = len(correct) / num_anomalies
                P.append(prec)
                R.append(rec)
                thresh += step
                thresh = round(thresh, 3)

            P = [P[0]] + P
            pr_auc = auc(R, P)
            try:
                plt.figure(figsize=[8, 6])
                plt.plot(R, P)
                plt.title('Precision Recall Curve  || auPR :' + "{:0.4f}".format(pr_auc), fontsize=15)
                plt.xlabel('Recall', fontsize=15)
                plt.ylabel('Precision', fontsize=15)
                plt.show()
            except:
                pass
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
    default=2,
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
    log_op = 'Mean AuPR over {} | {} | runs {:5f} Std {:.5f}'.format(num_runs, key, mean_all_runs, np.std(_aupr))
    LOGGER.info(log_op)
    LOGGER.info(' Details ' + str(_aupr))

utils.close_logger(LOGGER)