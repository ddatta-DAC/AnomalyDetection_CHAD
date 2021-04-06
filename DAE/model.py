
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('./..')
sys.path.append('./../..')
sys.path.append('./../../..')

import torch
from torch import FloatTensor as FT
from torch import LongTensor as LT
from torch import nn
from torch.nn import functional as F
import os
from collections import OrderedDict
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions.normal import Normal
import math

try:
    from data_fetcher import data_fetcher
except:
    from .data_fetcher import data_fetcher

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print('Current device  >> ', DEVICE)
print('=========================== ')


def get_Activation(act):
    if act == 'tanh': return nn.Tanh()
    if act == 'sigmoid': return nn.Sigmoid()
    if act == 'none': return nn.Identity()
    if act == 'relu': return nn.ReLU()
    return nn.ReLU()

0
class module_LPT_AE(nn.Module):
    def __init__(
            self,
            data_dim,
            layer_dims,  # Provide the half (encoder only)
            op_activation='sigmoid',
            layer_activation='sigmoid',
            dropout=0.2
    ):
        super(module_LPT_AE, self).__init__()
        self.module_encoder = nn.ModuleList()
        self.module_decoder = nn.ModuleList()
        self.num_trained = 0
        self.layer_dims = layer_dims
        self.op_activation = op_activation
        self.layer_activation = layer_activation
        self.dropout_rate = dropout
        self.data_dim = data_dim
        self.num_layers = len(layer_dims)
        self.mode = 'ae'  # options are ae, encoder

    # =====
    # Adds in 1 layer of AE
    # =====
    def add_layer(self, layer_idx):
        print('Adding layer index ', layer_idx)
        if layer_idx >= self.num_layers:
            exit(1)
        # add in encoder
        _add_dropout = True
        if layer_idx == 0:
            inp_dim = self.data_dim
        else:
            inp_dim = self.layer_dims[layer_idx - 1]

        _layers = []
        op_dim = self.layer_dims[layer_idx]
        _layers.append(nn.Linear(inp_dim, op_dim))
        if _add_dropout:
            _layers.append(nn.Dropout(self.dropout_rate))
        _layers.append(get_Activation(self.layer_activation))

        self.module_encoder.append(
            nn.Sequential(
              *_layers
            )
        )
        # Swap the values  for decoder
        inp_dim, op_dim = op_dim, inp_dim

        # Last layer
        if layer_idx == 0:
            act = self.op_activation
            _add_dropout = False
        else:
            _add_dropout = True
            act = self.layer_activation
        # Insert at start

        _layers = []
        _layers.append(nn.Linear(inp_dim, op_dim))
        if _add_dropout:
            _layers.append(nn.Dropout(self.dropout_rate))
        _layers.append( get_Activation(act))

        self.module_decoder.insert(
            0,
            nn.Sequential(
              *_layers
            )
        )
        return

    def forward(self, x):
        x1 = x
        for m in self.module_encoder:
            x1 = m(x1)
        z = x1
        x2 = x1
        for m in self.module_decoder:
            x2 = m(x2)

        if self.mode == 'ae':
            return z, x2
        elif self.mode == 'encoder':
            return z
        else:
            return x2

    # =========================
    # Return params by layer
    # =========================
    def get_trainable_layer_params(self, layer_idx=-1):
        if layer_idx == -1:
            return list(self.parameters())

        e = list(self.module_encoder[layer_idx].parameters())
        d = list(self.module_decoder[-(layer_idx + 1)].parameters())
        return e + d


class StackedAE():

    def __init__(
            self,
            device,
            data_dim,
            layer_dims,  # Provide the half (encoder only)
            op_activation='sigmoid',
            layer_activation='sigmoid',
            dropout=0.05,
            LR=0.05,
            num_epochs_1=10,
            num_epochs_2=25,
            min_epochs=10,
            batch_size=256,
            stop_threshold=0.001,
            checkpoint_dir=None,
            use_warm_start = False
    ):
        self.log_interval = 100
        self.device = device

        self.ae = module_LPT_AE(
            data_dim,
            layer_dims,
            op_activation,
            layer_activation,
            dropout
        )
        self.ae_num_layers = len(layer_dims)
        self.ae = self.ae.to(self.device)
        self.LR = LR
        self.min_epochs = min_epochs
        self.num_epochs_1 = num_epochs_1
        self.num_epochs_2 = num_epochs_2
        self.batch_size = batch_size
        self.latent_dim = layer_dims[-1]
        self.stop_threshold = stop_threshold
        self.max_patience = 5
        self.use_warm_start = use_warm_start
        # ---------------------
        # Dir to save the pre trained model results
        # ---------------------
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.FLAG_ae_setup = False
        self.break_threshold = 0.001
        return

    def pretrain_ae(self, data):
        print('Greedy layerwise pretarining started')
        self.ae.mode = 'ae'
        self.ae.train()

        num_epochs = self.num_epochs_1
        batch_size = self.batch_size
        log_interval = self.log_interval
        print(' Num AE layers ', self.ae_num_layers)

        for l_idx in range(self.ae_num_layers):
            print('Adding layer :', l_idx+1)
            self.ae.add_layer(l_idx)
            self.ae = self.ae.to(self.device)
            print('Current model ', self.ae)
            # train using the data
            params = self.ae.get_trainable_layer_params(layer_idx=l_idx)
            opt = torch.optim.Adam(
                params,
                lr=self.LR
            )

            for e in tqdm(range(num_epochs)):
                epoch_loss = []
                np.random.shuffle(data)
                num_batches = data.shape[0] // batch_size + 1
                for b_idx in range(num_batches):
                    opt.zero_grad()
                    x = data[b_idx * batch_size: (b_idx + 1) * batch_size]
                    x = FT(x).to(self.device)
                    
                    _, x_R = self.ae(x)
                    b_loss = F.mse_loss(
                        x_R, x, reduction='none' 
                    )
                    b_loss = torch.sum(b_loss,dim=-1,keepdim=False)
                    b_loss = torch.mean(b_loss,dim=0,keepdim=False)
                    b_loss.backward()
                    opt.step()
                    loss_val = b_loss.cpu().data.numpy()
                    epoch_loss.append(loss_val)
                    if b_idx % log_interval == 0:
                        print('Loss {:4f}'.format(loss_val))
                print(' Epoch {} loss {:4f}'.format(e + 1, np.mean(epoch_loss)))

        print('Greedy layer-wise pretraining [Done]')

        # =======================
        # Now train the entire autoencoder
        # =======================

        opt = torch.optim.Adam(
            list(self.ae.parameters()),
            lr=self.LR
        )

        for e in tqdm(range(num_epochs)):
            epoch_loss = []
            np.random.shuffle(data)
            num_batches = data.shape[0] // batch_size + 1
            
            for b_idx in range(num_batches):
                opt.zero_grad()
                x = data[b_idx * batch_size: (b_idx + 1) * batch_size]
                x = FT(x).to(self.device)
                _, x_R = self.ae(x)
                b_loss = F.mse_loss(
                    input=x, target=x_R, reduction='none'
                )
                b_loss = torch.sum(b_loss,dim=-1, keepdim=False)
                b_loss = torch.mean(b_loss,dim=-1, keepdim=False)
                b_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 2)
                opt.step()
                loss_val = b_loss.cpu().data.numpy()
                epoch_loss.append(loss_val)
                if b_idx % log_interval == 0:
                    print('Loss {:4f}'.format(loss_val))
            print(' Epoch {} loss {:4f}'.format(e + 1, np.mean(epoch_loss)))
        self.FLAG_ae_setup = True
        return

    # =====================================
    # main training function
    # =====================================
    def train_model(self, data ):
        use_warm_start = self.use_warm_start
        ae_weights_file = os.path.join(self.checkpoint_dir, 'ae_model.pt')
        if use_warm_start and os.path.exists(ae_weights_file):
            if self.FLAG_ae_setup is False:
                # Set up the structure first , then load the weights
                for l_idx in range(self.ae_num_layers):
                    self.ae.add_layer(l_idx)
                    self.ae = self.ae.to(self.device)
                checkpoint = torch.load(ae_weights_file)
                self.ae.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.pretrain_ae(data)

            # torch.save(self.ae.state_dict(), ae_weights_file)
            torch.save({
                'model_state_dict': self.ae.state_dict()
            }, ae_weights_file)
            self.FLAG_ae_setup = True

        # =======================
        self.ae.train()
        self.ae.mode = 'train'

        num_epochs = self.num_epochs_2
        batch_size = self.batch_size
        num_batches = data.shape[0] // batch_size + 1
        _params = list(self.ae.parameters())

        opt = torch.optim.Adam(
            _params,
            lr=self.LR
        )

        prev_epoch_loss_mean = 0
        patience = 0
        for e in tqdm(range(1, num_epochs + 1)):
            print('Epoch :: {}'.format(e))
            np.random.shuffle(data)
            # ------------
            # cluster assignment s_ij
            # ------------

            self.ae.mode = 'ae'
            epoch_loss = []

            for b in range(num_batches):
                # ==================
                # Step 1
                # ==================
                self.ae.train()
                opt.zero_grad()
                _x = data[b * batch_size: (b + 1) * batch_size]
                _x = FT(_x).to(self.device)
                _, x_r = self.ae(_x)

                # Calculate the reconstruction loss
                ae_loss = F.mse_loss(_x, x_r, reduction='none')
                ae_loss = torch.sum(ae_loss, dim=-1, keepdim=False)
                loss = torch.mean(ae_loss, dim=0)
                loss.backward()

                epoch_loss.append(loss.cpu().data.numpy())
                opt.step()
                if b % self.log_interval == 0:
                    print('Loss {:4f}'.format(loss.cpu().data.numpy()))

            epoch_loss_mean = np.mean(epoch_loss)
            diff = abs(epoch_loss_mean - prev_epoch_loss_mean)
            _break_flag = False
            if diff < self.break_threshold and e > self.min_epochs:
                if patience > self.max_patience :
                    _break_flag = True
                else:
                    patience += 1
            else:
                # Reset patience
                patience = 0
            prev_epoch_loss_mean = epoch_loss_mean
            if _break_flag:
                print('Breaking training loss staying same ')
                patience = 0
                break

        return

    # ========================
    # Score a single sample
    # ========================

    def __score_sample(self, x):
        self.ae.eval()
        self.ae.mode = 'ae'
        _, x_r = self.ae(x)
        recons_err = F.mse_loss(x, x_r, reduction='none')
        recons_err = torch.sum(recons_err, dim=-1, keepdim=False)
        return recons_err

    def score_samples(self, data):
        bs = self.batch_size
        num_batches = data.shape[0] // bs + 1
        res = []

        for b in tqdm(range(num_batches)):
            x = data[b * bs: (b + 1) * bs]
            x = FT(x).to(self.device)
            if x.shape[0] == 0:
                break
            r = self.__score_sample(x)
            r = r.cpu().data.numpy()
            res.extend(r)
        res = np.array(res)
        return res


