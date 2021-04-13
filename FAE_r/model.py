import pandas as pd
import numpy as np
import torch
from torch.nn import Module
from torch.nn import ModuleList
from torch import nn
from torch.nn import functional as F
import os
from collections import OrderedDict
import math
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions.normal import Normal
import math
from torch import FloatTensor as FT
from torch import LongTensor as LT
EPSILON = math.pow(10, -6)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print('Current device  >> ', DEVICE)
print('=========================== ')


def get_Activation(act):
    if act == 'tanh': return nn.Tanh()
    if act == 'sigmoid': return nn.Sigmoid()
    if act == 'none': return nn.Identity()
    return nn.ReLU()


class AE_encoder(nn.Module):
    def __init__(
            self,
            device,
            structure_config,
            dropout=0.1
    ):
        super(AE_encoder, self).__init__()
        self.device = device
        self.structure_config = structure_config

        # ================
        # Assume discrete layers are at the start
        # Everything is is in 1-0 form
        # ===========

        # Dictionary < field : dims >
        self.discrete_dims = OrderedDict(structure_config['discrete_dims'])
        self.real_dims = structure_config['real_dims']

        conatenated_inp_dim = 0
        encoder_discrete_xform = structure_config['encoder_discrete_xform']
        self.input_x_form_layers = nn.ModuleList()
        for column, dim in self.discrete_dims.items():
            if encoder_discrete_xform is not None and column in encoder_discrete_xform.keys():
                _fcn_ = encoder_discrete_xform[column]['dims']
                _act = encoder_discrete_xform[column]['activation']
                _layers = []
                inp = dim
                for k in _fcn_:
                    _layers.append(nn.Linear(inp, k))
                    _layers.append(get_Activation(_act))
                    inp = k
                conatenated_inp_dim += inp  # Last output
                self.input_x_form_layers.append(nn.Sequential(*_layers))
            else:
                self.input_x_form_layers.append(nn.Identity())
                # Handle binary case
                if dim == 2:
                    dim = 1
                conatenated_inp_dim += dim

        encoder_real_xform_dims = structure_config['encoder_real_xform']['dims']

        if encoder_real_xform_dims is not None and len(encoder_real_xform_dims) > 0:
            _act = structure_config['encoder_real_xform']['activation']
            _layers = []
            inp = self.real_dims
            for k in encoder_real_xform_dims:
                _layers.append(nn.Linear(inp, k))
                _layers.append(get_Activation(_act))
                inp = k
            self.input_x_form_layers.append(nn.Sequential(*_layers))
            conatenated_inp_dim += inp  # Last output
        else:
            conatenated_inp_dim += self.real_dims
            self.input_x_form_layers.append(nn.Identity())

        # ====
        # Put the concatenated (xformed) input through FCN
        # ====
        encoder_FCN_to_latent_dims = structure_config['encoder_FCN_to_latent']['dims']
        _act = structure_config['encoder_FCN_to_latent']['activation']
        self.ae_latent_dimension = structure_config['ae_latent_dimension']

        _layers = []
        inp = conatenated_inp_dim
        for k in encoder_FCN_to_latent_dims + [self.ae_latent_dimension]:
            _layers.append(nn.Linear(inp, k))
            _layers.append(nn.Dropout(dropout))
            _layers.append(get_Activation(_act))
            inp = k
        self.FC_z = nn.Sequential(*_layers)

        # ====================
        input_split_schema = []
        for val in list(self.discrete_dims.values()):
            if val == 2:
                input_split_schema.append(1)
            else:
                input_split_schema.append(val)

        self.input_split_schema = input_split_schema + [self.real_dims]
        print('split schema ', self.input_split_schema)
        return

    def forward(self, X):
        X = X.squeeze(1)
        #         print('encoder x', X.shape)

        split_X = torch.split(X, self.input_split_schema, dim=1)
        res = []
        for i in range(len(split_X)):
            res.append(self.input_x_form_layers[i](split_X[i]))
        x_concat = torch.cat(res, dim=1)
        op = self.FC_z(x_concat)
        return op


class AE_decoder(nn.Module):
    def __init__(
            self,
            device,
            structure_config=None,
            dropout=0.1
    ):
        super(AE_decoder, self).__init__()
        self.device = device
        self.structure_config = structure_config
        self.discrete_dims = OrderedDict(structure_config['discrete_dims'])
        self.real_dims = structure_config['real_dims']
        total_op_dim = 0
        for val in self.discrete_dims.values():
            # Handle Binary case
            if val == 2:
                val = 1
            total_op_dim += val
        total_op_dim += self.real_dims
        # =====================
        # IF no projection is wanted
        # ======================
        self.PROJECTION = False
        latent_dim = structure_config['ae_latent_dimension']

        inp_dim = latent_dim
        fcn_dims = structure_config['decoder_FC_from_latent']['dims']
        _act = structure_config['decoder_FC_from_latent']['activation']
        _layers = []

        for k in fcn_dims:
            _layers.append(nn.Linear(inp_dim, k))
            _layers.append(nn.Dropout(dropout))
            _layers.append(get_Activation(_act))
            inp_dim = k

        _layers.append(nn.Linear(inp_dim, total_op_dim))
        _layers.append(nn.Sigmoid())
        self.FC_z = nn.Sequential(*_layers)
        # the output size is inp_dim
        return

    def forward(self, z):

        z1 = self.FC_z(z)
        if self.PROJECTION:
            res = []
            for module in self.module_list:
                res.append(module(z1))
            res = torch.cat(res, dim=1)
            return res
        return z1


class AE(nn.Module):

    def __init__(
            self,
            device,
            encoder_structure_config,
            decoder_structure_config,
            latent_dim,
            dropout=0.1
    ):
        super(AE, self).__init__()
        self.device = device
        self.encoder = AE_encoder(
            device,
            encoder_structure_config,
            dropout
        )
        self.decoder = AE_decoder(
            device,
            decoder_structure_config,
            dropout
        )

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.mode = None
        return

    def forward(self, x):
        z = self.encoder(x)

        if self.mode == 'compress':
            return z
        x_recon = self.decoder(z)
        return x_recon, z


class AE_loss_module(nn.Module):
    def __init__(
            self,
            device,
            structure_config=None
    ):
        super(AE_loss_module, self).__init__()
        self.device = device
        self.structure_config = structure_config
        num_fields = len(structure_config['discrete_dims']) + 1
        # ===========
        # config Format :
        # decoder output dim , loss type , data type
        # decoder output dim is the number of categories for onehot data
        # ===========

        # self.loss_FC = nn.Linear(num_fields, 1, bias=False)
        print('Loss structure config', structure_config)
        self.discrete_dims = self.structure_config['discrete_dims']

        real_dims = self.structure_config['real_dims']
        split_schema = []
        for column, dim in self.discrete_dims.items():
            if dim == 2:  # Binary case
                dim = 1
            split_schema.append(dim)
        split_schema.append(real_dims)
        print(' Loss module split schema ', split_schema)
        self.split_schema = split_schema
        self.real_dims = real_dims
        return

    def forward(
            self,
            x_true,
            x_pred
    ):
        # ------------------
        # Split the x
        # ------------------
        x_true = x_true.to(self.device)
        x_pred = x_pred.to(self.device)
        mse = F.mse_loss(x_true, x_pred, reduction='none')
        # mse = torch.sum(mse, dim=1, keepdim=False)
        return mse

# ==================== #

class model_FAER(nn.Module):
    def __init__(
            self,
            device,
            latent_dim,
            encoder_structure_config,
            decoder_structure_config,
            loss_structure_config,
            dropout
    ):
        super(model_FAER, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.ae_module = AE(
            device=device,
            latent_dim=latent_dim,
            encoder_structure_config=encoder_structure_config,
            decoder_structure_config=decoder_structure_config,
            dropout=dropout
        )
        self.ae_module = self.ae_module.to(self.device)
        self.ae_loss_module = AE_loss_module(self.device, loss_structure_config)
        self.ae_loss_module = self.ae_loss_module.to(self.device)

        self.num_fields = len(encoder_structure_config['discrete_dims']) + 1
        self.latent_dim = self.ae_module.encoder.ae_latent_dimension

        self.mode = 'train'
        return

    def score_samples(self, x_true):
        x_pred, _ = self.ae_module(x_true)
        # ---------------
        # Calculate reconstruction loss
        # ---------------
        recons_loss = self.ae_loss_module(
            x_true,
            x_pred
            )
        recons_loss = torch.sum(
            recons_loss,
            dim=1,
            keepdim=False
        ).to(self.device)

        return recons_loss


    def forward(
            self,
            x
    ):
        global EPSILON
        # ================================================ #
        # Training mode
        # ================================================ #
        if self.mode == 'train':
            # Return the per sample loss
            x_recon, z = self.ae_module(x)
            loss_md = self.ae_loss_module(
                x_true=x,
                x_pred=x_recon
            )
            # loss_md (multidimensional)  has shape [ batch, num_fields ]
            # Sample loss is sum of the loss pertaining to all the fields
            sample_loss = torch.sum(
                loss_md,
                dim=1,
                keepdim=True
            ).to(self.device)

            return sample_loss
        # ================================================ #

        elif self.mode == 'test':
            recons_loss = self.score_samples(x)
            return recons_loss


# ====================================================
# Main class
# ====================================================

class model_FAER_container():
    def __init__(
            self,
            device,
            latent_dim,
            encoder_structure_config,
            decoder_structure_config,
            loss_structure_config,
            optimizer='Adam',
            batch_size=256,
            num_epochs=20,
            learning_rate=0.05,
            log_interval=100,
            dropout = 0.1
    ):

        self.device = device
        self.log_interval = log_interval
        self.LR = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.network_module = model_FAER(
            device,
            latent_dim,
            encoder_structure_config,
            decoder_structure_config,
            loss_structure_config,
            dropout=dropout
        )
        return

    def train_model(
            self,
            X
    ):
        self.network_module.ae_module.mode = 'train'
        self.network_module.ae_module.train()
        self.network_module.ae_loss_module.train()
        self.network_module.mode = 'train'
        learning_rate = self.LR
        parameters = list(self.network_module.parameters())
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate
        )
        log_interval = self.log_interval
        losses = []
        bs = self.batch_size

        for epoch in tqdm(range(1, self.num_epochs + 1)):
            t = epoch
            epoch_losses = []
            num_batches = X.shape[0] // bs + 1
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            X_P = X[idx]
            X_P = FT(X_P).to(self.device)

            for b in range(num_batches):
                # self.network_module.zero_grad()
                self.optimizer.zero_grad()
                _x_p = X_P[b * bs: (b + 1) * bs]
                batch_loss = self.network_module(_x_p)

                # Standard AE loss
                batch_loss = batch_loss.squeeze(1)
                batch_loss = torch.mean(batch_loss, dim=0, keepdim=False)
                # ====================
                # Clip Gradient
                # ====================
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network_module.parameters(), 2)
                self.optimizer.step()

                loss_value = batch_loss.clone().cpu().data.numpy()
                losses.append(loss_value)
                if b % log_interval == 0:
                    print(' Epoch {} Batch {} Loss {:.4f} '.format(
                        epoch,
                        b,
                        batch_loss
                    )
                    )

                epoch_losses.append(loss_value)
            print('Epoch loss ::', np.mean(epoch_losses))
        self.network_module.mode = 'test'
        return epoch_losses

    def get_compressed_embedding(
            self,
            data
    ):
        self.network_module.eval()
        self.network_module.mode = 'test'
        self.network_module.ae_module.mode = 'compress'
        X = FT(data).to(self.device)
        bs = 500
        num_batches = data.shape[0] // self.batch_size + 1
        output = []
        for b in range(num_batches):
            _x = X[b * bs: (b + 1) * bs]
            z = self.network_module.ae_module(_x)
            z_data = z.clone().cpu().data.numpy()
            output.extend(z_data)
        return output

    # ================================================ #
    # Score the samples
    # ================================================ #
    def get_score(
            self,
            data
    ):
        self.network_module.eval()
        self.network_module.mode = 'test'
        self.network_module.ae_module.mode = 'test'
        X = FT(data).to(self.device)
        bs = 500
        num_batches = data.shape[0] // self.batch_size + 1
        output = []
        for b in range(num_batches):
            _x = X[b * bs: (b + 1) * bs]
            if _x.shape[0] == 0: continue
            z = self.network_module(_x)
            z_data = z.clone().cpu().data.numpy()
            output.extend(z_data)
        return output