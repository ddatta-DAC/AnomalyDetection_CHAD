import json
import torch
import sys
sys.path.append('')
sys.path.append('./..')
sys.path.append('./networks')
import os
import numpy as np
from tqdm import tqdm

try:
    from .networks.AE import FC_dec
    from .networks.AE import FC_enc
except:
    from networks.AE import FC_dec
    from networks.AE import FC_enc

try:
    from networks.main import build_network
except:
    from .networks.main import build_network

try:
    from optim.deepSVDD_trainer import DeepSVDDTrainer
except:
    from .optim.deepSVDD_trainer import DeepSVDDTrainer

# try:
#     from optim.ae_trainer import AETrainer
# except:
#     from .optim.ae_trainer import AETrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device ::', DEVICE)


class DeepSVDD(object):

    def __init__(self, DEVICE, objective: str = 'soft-boundary', nu: float = 0.1 ):
        self.device = DEVICE
        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.R = 0.0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network \phi
        self.net_dec = None
        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None


    def set_network(self, fc_layer_dims):
        self.net = FC_enc(fc_layer_dims)
        # Pretraining using AE requires inverted FC
        self.net_dec = FC_dec(fc_layer_dims[::-1])
        print(self.net)
        print(self.net_dec)
        
    def train(
            self,
            train_X,
            LR=0.001,
            num_epochs=50,
            batch_size=256,
            warm_up_epochs = 10,
            ae_epochs = 10
    ):
        self.trainer = DeepSVDDTrainer(
            self.device,
            objective=self.objective,
            R=self.R,
            c=self.c,
            nu=self.nu,
            LR=LR,
            num_epochs=num_epochs,
            batch_size=batch_size,
            warm_up_epochs = warm_up_epochs,
            ae_epochs = ae_epochs
        )
        # ------------------------------- #
        # Get the model
        self.net = self.trainer.train(self.net, self.net_dec, train_X)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list

    def test(
            self,
            test_X
    ):
        """Tests the Deep SVDD model on the test data."""
        scores = self.trainer.test(test_X)
        return scores




