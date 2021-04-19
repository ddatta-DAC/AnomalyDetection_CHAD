import sys
sys.path.append('./..')
sys.path.append('./../..')
import argparse
from sklearn.metrics import auc
import pandas as pd
from torch import FloatTensor as FT
import numpy as np
from tqdm import tqdm
from pprint import pprint
import torch
import yaml
import matplotlib.pyplot  as plt
from torch import FloatTensor as FT
from torch.autograd import Variable
import matplotlib.pyplot as plt
try:
    from .data_fetcher import data_fetcher
except:
    from data_fetcher import data_fetcher
try:
    from .DAGMM import DaGMM
except:
    from DAGMM import DaGMM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Current device  >> ', DEVICE)
try:
    from . import utils
except:
    import utils

ID_COL = 'PanjivaRecordID'
# =====================================
def create_config(
        data_set
):
    # Should return :
    # data_dict
    # meta_data_df [column, dimension]
    global ID_COL
    config_file = 'config.yaml'
    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)

    data_dict, meta_data_df = data_fetcher.get_data(data_set, one_hot=True)

    # discrete_columns : { column_name : num_categories }
    discrete_column_dims = {
        k: v for k, v in
        zip(list(meta_data_df['column']), list(meta_data_df['dimension']))
    }

    num_discrete_columns = 0
    for column, dim in discrete_column_dims.items():
        if dim == 2:
            num_discrete_columns += 1
        else:
            num_discrete_columns += dim

    num_real_columns = len(data_dict['train'].columns) - num_discrete_columns
    if ID_COL in data_dict['train'].columns:
        num_real_columns = num_real_columns - 1

    print('Num real columns :: ', num_real_columns)
    print('Num discrete columns ::', num_discrete_columns)
    latent_dim = config['ae_latent_dimension']

    encoder_structure_config = {}
    encoder_structure_config['discrete_column_dims'] = discrete_column_dims
    encoder_structure_config['num_discrete'] = num_discrete_columns
    encoder_structure_config['num_real'] = num_real_columns
    encoder_structure_config['encoder_layers'] = {
        'activation': config[data_set]['encoder_layers']['activation'],
        'layer_dims': config[data_set]['encoder_layers']['layer_dims'] + [latent_dim]
    }

    # ======================================================
    # Set decoder structure
    # =========

    decoder_structure_config = {}
    final_op_dims = num_real_columns
    for k, v in discrete_column_dims.items():
        if v == 2:
            v = 1
        final_op_dims += v

    decoder_structure_config['discrete_column_dims'] = discrete_column_dims
    decoder_structure_config['num_discrete'] = num_discrete_columns
    decoder_structure_config['num_real'] = num_real_columns
    decoder_structure_config['decoder_layers'] = {
        'activation': config[data_set]['decoder_layers']['activation'],
        'layer_dims': [latent_dim] + config[data_set]['decoder_layers']['layer_dims'] + [final_op_dims]
    }
    decoder_structure_config['final_output_dim'] = final_op_dims

    # =====================
    # GMM
    # =====================
    gmm_input_dims = latent_dim + 2
    activation = config[data_set]['gmm']['FC_layer']['activation']
    num_components = config[data_set]['gmm']['num_components']
    FC_layer_dims = [gmm_input_dims] + config[data_set]['gmm']['FC_layer']['dims'] + [num_components]
    FC_dropout = config[data_set]['gmm']['FC_dropout']
    gmm_structure_config = {
        'num_components': num_components,
        'FC_layer_dims': FC_layer_dims,
        'FC_dropout': FC_dropout,
        'FC_activation': activation

    }
    loss_structure_config = []

    for column, dim in discrete_column_dims.items():
        loss_structure_config.append(
            {
                'dim': dim,
                'type': 'onehot'
            }
        )
    loss_structure_config.append(
        {
            'dim': num_real_columns,
            'type': 'real'
        }
    )


    return encoder_structure_config, decoder_structure_config, gmm_structure_config, loss_structure_config, latent_dim


def train(
        dagmm_obj,
        data,
        num_epochs=100,
        batch_size=256,
        LR=0.001
):
    optimizer = torch.optim.Adam(dagmm_obj.parameters(), lr=LR)
    dagmm_obj.train()
    log_interval = 50

    for epoch in tqdm(range(num_epochs)):
        num_batches = data.shape[0] // batch_size + 1
        epoch_losses = []
        np.random.shuffle(data)
        X = FT(data).to(DEVICE)
        lambda_energy = 0.1
        lambda_cov_diag = 0.005
        for b in range(num_batches):
            optimizer.zero_grad()
            input_data = X[b * batch_size: (b + 1) * batch_size]
            enc, dec, z, gamma = dagmm_obj(input_data)
            total_loss, sample_energy, recon_error, cov_diag = dagmm_obj.loss_function(
                input_data, dec, z, gamma,
                lambda_energy,
                lambda_cov_diag
            )

            dagmm_obj.zero_grad()
            total_loss = Variable(total_loss, requires_grad=True)
            total_loss.backward()
            epoch_losses.append(total_loss.cpu().data.numpy())
            torch.nn.utils.clip_grad_norm_(dagmm_obj.parameters(), 5)
            optimizer.step()

            loss = {}
            loss['total_loss'] = total_loss.data.item()
            loss['sample_energy'] = sample_energy.item()
            loss['recon_error'] = recon_error.item()
            loss['cov_diag'] = cov_diag.item()

            if (b + 1) % log_interval == 0:
                log = ' '
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
        print('Epoch loss :: {:.4f}'.format(np.mean(epoch_losses)))
    return dagmm_obj


def test(
        dagmm_obj,
        data_dict
):
    global DEVICE
    global ID_COL

    print("======================TEST MODE======================")
    dagmm_obj.eval()
    N = 0
    mu_sum = 0
    cov_sum = 0
    gamma_sum = 0
    train_df = data_dict['train']
    try:
        del train_df[ID_COL]
    except:
        pass
    train_X = train_df.values
    batch_size = 507
    num_batches = train_X.shape[0] // batch_size + 1
    for b in range(num_batches):
        input_data = train_X[b * batch_size: (b + 1) * batch_size]
        input_data = FT(input_data).to(DEVICE)
        enc, dec, z, gamma = dagmm_obj(input_data)
        phi, mu, cov = dagmm_obj.compute_gmm_params(z, gamma)
        batch_gamma_sum = torch.sum(gamma, dim=0)
        gamma_sum += batch_gamma_sum
        mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
        cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only
        N += input_data.size(0)

    train_phi = gamma_sum / N
    train_mu = mu_sum / gamma_sum.unsqueeze(-1)
    train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

    print("N:", N)
    print("phi :", train_phi)
    print("mu : ", train_mu)
    print("covariance :", train_cov)

    # ============================ #
    # Get sample energy for test set
    # ============================ #
    test_energy = []
    train_labels = []
    train_z = []

    test_df = data_dict['test']
    try:
        del test_df[ID_COL]
    except:
        pass
    test_X = test_df.values

    num_batches = test_X.shape[0] // batch_size + 1
    print('Size of test ', test_X.shape[0])
    for b in range(num_batches):
        input_data = test_X[b * batch_size: (b + 1) * batch_size]
        input_data = FT(input_data).to(DEVICE)
        enc, dec, z, gamma = dagmm_obj(input_data)
        sample_energy, cov_diag = dagmm_obj.compute_energy(
            z,
            phi=train_phi,
            mu=train_mu,
            cov=train_cov,
            size_average=False
        )
        test_energy.append(sample_energy.data.cpu().numpy())
        # train_z.append(z.data.cpu().numpy())
        # train_labels.append(labels.numpy())

    test_energy = np.concatenate(test_energy, axis=0)
    print('test_energy', test_energy.shape)
    test_labels = [0 for _ in range(test_X.shape[0])]
    auc_list = []

    # ===========
    # Get per sample energy of the anomalies
    # ===========
    num_anomaly_sets = 5

    auc_result = {}

    for anomaly_key in ['anom_2_']:
        auc_list = []
        for idx in range(num_anomaly_sets):
            key = 'anom_' + str(idx)
            anom_df = data_dict[key]
            try:
                del anom_df[ID_COL]
            except:
                pass
            anom_X = anom_df.values

            anom_labels = [1 for _ in range(anom_X.shape[0])]
            anom_energy = []
            num_batches = anom_X.shape[0] // batch_size + 1

            for b in range(num_batches):
                input_data = anom_X[b * batch_size: (b + 1) * batch_size]
                input_data = FT(input_data).to(DEVICE)
                enc, dec, z, gamma = dagmm_obj(input_data)
                sample_energy, cov_diag = dagmm_obj.compute_energy(
                    z,
                    phi=train_phi,
                    mu=train_mu,
                    cov=train_cov,
                    size_average=False
                )
                anom_energy.append(sample_energy.data.cpu().numpy())

            anom_energy = np.concatenate(anom_energy, axis=0)

            combined_energy = np.concatenate([anom_energy, test_energy], axis=0)
            combined_labels = np.concatenate([anom_labels, test_labels], axis=0)

            res_data = []
            for i, j in zip(combined_energy, combined_labels):
                res_data.append((i, j))
            res_df = pd.DataFrame(res_data, columns=['score', 'label'])

            #  Normalize values
            def _normalize_(val, _min, _max):
                return (val - _min) / (_max - _min)

            _max = max(combined_energy)
            _min = min(combined_energy)

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

            while thresh > _min:
                sel = res_df.loc[res_df['score'] >= thresh]
                if len(sel) == 0:
                    thresh -= step
                    continue
                correct = sel.loc[sel['label'] == 1]
                prec = len(correct) / len(sel)
                rec = len(correct) / num_anomalies
                P.append(prec)
                R.append(rec)
                thresh -= step
            P = [P[0]] + P

            pr_auc = auc(R, P)
            auc_list.append(pr_auc)
            print("AUC : {:0.5f} ".format(pr_auc))

        mean_auc = np.mean(auc_list)
        print(' (Mean) AUC {:0.5f} '.format(mean_auc))
        auc_result[anomaly_key] = mean_auc
    return auc_result


def execute_run(DATA_SET):
    global LOGGER
    global ID_COL

    config_file = 'config.yaml'
    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)
    batch_size = config['batch_size']
    anomaly_ratio = config['anomaly_ratio']
    learning_rate = config['LR']

    data_dict, _ = data_fetcher.get_data(
        DATA_SET,
        one_hot=True,
        num_anom_sets=5,
        anomaly_ratio=anomaly_ratio
    )

    train_df = data_dict['train']
    try:
        del train_df[ID_COL]
    except:
        pass
    train_X = train_df.values
    encoder_structure_config, decoder_structure_config, gmm_structure_config, _, latent_dim = create_config(
        DATA_SET
    )
    pprint(encoder_structure_config)
    pprint(decoder_structure_config)

    dagmm_obj = DaGMM(
        DEVICE,
        encoder_structure_config,
        decoder_structure_config,
        n_gmm = gmm_structure_config['num_components'],
        ae_latent_dim=latent_dim
    )
    dagmm_obj = dagmm_obj.to(DEVICE)
    print(dagmm_obj)

    dagmm_obj = train(
        dagmm_obj,
        train_X,
        num_epochs=400,
        batch_size=batch_size,
        LR=0.0001
    )
    mean_aupr, std = test(
        dagmm_obj,
        data_dict
    )

    return  mean_aupr, std

# ========================================== #


parser = argparse.ArgumentParser(description='Run the model ')
parser.add_argument(
    '--DATA_SET',
    type=str,
    help=' Which data set ?',
    default=None,
    choices=['us_import1', 'us_import2', 'us_import3', 'nb15', 'gureKDD']
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
results = []
for n in range(1,num_runs+1):
    mean_aupr, std = execute_run(DATA_SET)
    results.append(mean_aupr)
    LOGGER.info(' Run {}: Mean: {:4f} | Std {:4f}'.format(n,mean_aupr,std))
mean_all_runs = np.mean(results)
print('Mean AuPR over  {} runs {:4f}'.format(num_runs, mean_all_runs))
print('Details: ', results)

LOGGER.info('Mean AuPR over  {} runs {:4f} Std {:4f}'.format(num_runs, mean_all_runs, np.std(results)))
LOGGER.info(' Details ' + str(results))
utils.close_logger(LOGGER)
