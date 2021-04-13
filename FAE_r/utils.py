import numpy as np
import sys
import os
sys.path.append('./..')
sys.path.append('./../..')
from collections import OrderedDict
import pickle
import yaml
import matplotlib.pyplot  as plt
from sklearn.metrics import auc
import logging
import logging.handlers
from time import time
from datetime import datetime
try:
    from data_fetcher import data_fetcher
except:
    from .data_fetcher import data_fetcher

# ----
# First one normal
# Second one anomalies
# ----
domain_dims = None
ID_COL = 'PanjivaRecordID'

def get_domain_dims(DIR):
    global domain_dims
    with open(os.path.join('./../generated_data_v1/', DIR,  'domain_dims.pkl'), 'rb') as fh:
        domain_dims = OrderedDict(pickle.load(fh))
    return


def get_logger(LOG_FILE):
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


def create_config(data_set):
    # Should return :
    # data_dict
    # meta_data_df [column, dimension]
    global domain_dims
    global ID_COL
    get_domain_dims(data_set)
    config_file = 'architecture_config.yaml'
    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)


    latent_dim = config['ae_latent_dimension']
    data_dict, meta_data_df = data_fetcher.get_data(data_set=data_set, one_hot=True)

    # discrete_columns : { column_name : num_categories }
    discrete_dims = domain_dims
    _df_ = data_dict['train']
    del _df_[ID_COL]
    count_discrete_dims = sum(domain_dims.values())
    real_dims = len(_df_.columns) - count_discrete_dims

    # ---------------
    # encoder_structure_config['ip_layers']
    # Format :
    # [ 'emb|onehot', num_categories, [ embedding dimension ]
    # ---------------
    encoder_structure_config = {
        'real_dims': real_dims,
        'discrete_dims': discrete_dims,
        'encoder_FCN_to_latent': config['encoder_FCN_to_latent'],
        'ae_latent_dimension': config['ae_latent_dimension'],
        'encoder_discrete_xform': config['encoder_discrete_xform'],
        'encoder_real_xform': config['encoder_real_xform']
    }

    # ======================================================
    # Set decoder structure
    # =========

    decoder_structure_config = {
        'real_dims': real_dims,
        'discrete_dims': discrete_dims,
        'decoder_FC_from_latent': config['decoder_FC_from_latent'],
        'decoder_discrete_xform': config['decoder_discrete_xform'],
        'decoder_real_xform': config['decoder_real_xform'],
        'ae_latent_dimension': config['ae_latent_dimension']
    }

    # ================
    # Format decoder_field_layers:
    # { idx : [[dim1,dim2], op_activation ]
    # ================
    loss_structure_config = {
        'discrete_dims': discrete_dims,
        'real_loss_func': config['real_loss_func'],
        'real_dims': real_dims
    }

    return encoder_structure_config, decoder_structure_config, loss_structure_config, latent_dim
