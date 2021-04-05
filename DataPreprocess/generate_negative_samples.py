import pandas as pd
import os
import sys
import numpy as np
import pickle
import yaml
from joblib import Parallel, delayed
from sklearn.preprocessor import OneHotEncoder
from sklearn.compose import ColumnTransformer
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import argparse


domain_dims = None
DIR = None
CONFIG = None
CONFIG_FILE = 'config.yaml'
ID_COL = 'PanjivaRecordID'
categorical_columns = None
use_cols = None
freq_bound = None
save_dir = None
categorical_columns = None
numeric_columns = None


def get_domain_dims():
    global DIR
    global save_dir
    global domain_dims
    with open(os.path.join(save_dir, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)
    return


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
    global ID_COL
    global numeric_columns
    global categorical_columns, numeric_columns

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
    save_dir = CONFIG['save_dir']

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(
        CONFIG['save_dir'],
        DIR
    )
    DATA_SOURCE = save_dir

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    use_cols = [ID_COL] + categorical_columns + numeric_columns
    freq_bound_PERCENTILE = CONFIG['freq_bound_PERCENTILE']
    freq_bound_ABSOLUTE = CONFIG['freq_bound_ABSOLUTE']
    column_value_filters = CONFIG[DIR]['column_value_filters']

    _cols = list(use_cols)
    _cols.remove(ID_COL)
    get_domain_dims()
    return


def aux_gen(
        row,
        column_encoder,
        num_samples=10
):
    global categorical_columns, numeric_columns
    row_vals = row.values
    num_cat = len(categorical_columns)
    num_real_dims = len(numeric_columns)
    real_part = row_vals[-num_real_dims:]
    cat_part = row_vals[:num_cat]
    ns = num_samples

    # ======
    a = min(np.random.randint(1, num_real_dims // 2), num_real_dims)
    b = min(np.random.randint(1, num_real_dims // 2), num_real_dims)

    c = num_real_dims - (a + b)
    # Adding -.5 to shift noise to be between -.5 to .5
    noise = np.concatenate(
        [np.random.random_sample([ns, a]) - 0.5,
         np.random.random_sample([ns, b]) + 0.5,
         np.zeros([ns, c])],
        axis=1
    )

    for i in range(ns):
        np.random.shuffle(noise[i])
    # ---
    # noise shape [ ns, num_real_dims ]

    part_r_duplicated = np.tile(real_part, ns).reshape([ns, num_real_dims])
    part_r_duplicated = part_r_duplicated + noise

    # ------------------------------
    # For categorical variables
    # ------------------------------

    P = [np.power(_ / sum(categorical_columns), 0.75) for _ in categorical_columns]
    P = [_ / sum(P) for _ in P]
    part_c_duplicated = np.tile(cat_part, ns).reshape([ns, num_cat])

    for i in range(ns):
        _copy = np.array(row_vals)[:num_cat]
        if num_cat < 3:
            pert_idx = np.random.choice(list(np.arange(num_cat)), size=1, replace=False, p=P)
        else:
            pert_idx = np.random.choice(
                list(np.arange(num_cat)),
                size=np.random.randint(1, num_cat // 2 + 1),
                replace=False,
                p=P
            )

        for j in pert_idx:
            _attr = categorical_columns[j]
            _copy[j] = np.random.choice(
                np.arange(domain_dims[_attr], dtype=int), 1
            )
        part_c_duplicated[i] = _copy

    _samples = np.concatenate([part_c_duplicated, part_r_duplicated], axis=1)
    row_vals = np.reshape(row.values, [1, -1])

    samples = np.concatenate([row_vals, _samples], axis=0)
    sample_cat_part = samples[:, :num_cat]
    samples_real_part = samples[:, -num_real_dims:]

    # =========================
    # Do a 1-hot transformation
    # Drop binary columns
    # =========================

    onehot_xformed = column_encoder.fit_transform(
        sample_cat_part
    )
    onehot_xformed = onehot_xformed.astype(np.int)
    print('>>> 1-0 part ', onehot_xformed.shape)
    samples = np.concatenate([onehot_xformed, samples_real_part], axis=1)
    pos = samples[0]
    neg = samples[1:]
    return pos, neg


def generate_pos_neg_data(
        train_df,
        num_samples=10
):
    global ID_COL
    global domain_dims
    try:
        del train_df[ID_COL]
    except:
        pass

    try:
        del train_df['label']
    except:
        pass

    num_cat = len(domain_dims)
    num_real = len(train_df.columns) - num_cat

    oh_encoder_list = []
    idx = 0
    for _, dim in domain_dims.items():
        name = "oh_" + str(idx)
        oh_encoder = OneHotEncoder(
            np.reshape(list(range(dim)), [1, -1]),
            sparse=False,
            drop=False
        )
        oh_encoder_list.append((name, oh_encoder, [idx]))
        idx += 1
    column_encoder = ColumnTransformer(
        oh_encoder_list
    )

    discrete_dim_list = list(domain_dims.values())
    n_jobs = mp.cpu_count()

    res = Parallel(n_jobs)(delayed(aux_gen)(
        row, discrete_dim_list, num_real, column_encoder, num_samples
    ) for i, row in tqdm(train_df.iterrows(), total=train_df.shape[0])
                           )
    pos = []
    neg = []
    for r in res:
        pos.append(r[0])
        neg.append(r[1])

    pos = np.array(pos)
    neg = np.array(neg)

    return pos, neg



def generate(
        num_neg_samples=10
):
    # Save files
    global save_dir, DIR, domain_dims

    train_df = pd.read_csv(save_dir,'train_scaled.csv',index_col=None)
    result_save_dir = os.path.join('model_data')
    Path(result_save_dir).mkdir(exist_ok=True,parents=True)

    path_obj = Path(save_dir)
    path_obj.mkdir(
        exist_ok=True, parents=True
    )
    pos_file_path = os.path.join(result_save_dir, 'pos_samples.npy')
    neg_file_path = os.path.join(result_save_dir, 'neg_samples.npy')

    if os.path.exists(pos_file_path) and os.path.exists(neg_file_path):
        print('Files exist!')

    pos, neg = generate_pos_neg_data(
        train_df,
        num_samples=num_neg_samples
    )
    neg = np.reshape(neg, [pos.shape[0], num_neg_samples, pos.shape[1]])
    print(pos.shape, neg.shape)
    np.save(pos_file_path, pos)
    np.save(neg_file_path, neg)

    return


parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5' , 'us_import6'],
    default= None
)

args = parser.parse_args()
DIR = args.DIR
set_up_config(args.DIR)
generate()