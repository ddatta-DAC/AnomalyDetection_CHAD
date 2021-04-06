import sys
import os
import pandas as pd
import numpy as np

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
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions.normal import Normal
import math
from sklearn.cluster import MiniBatchKMeans, KMeans

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


class module_LPT_AE(nn.Module):
    def __init__(
            self,
            data_dim,
            layer_dims,  # Provide the half (encoder only)
            op_activation='sigmoid',
            layer_activation='sigmoid',
            dropout=0.05
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
        if layer_idx == 0:
            inp_dim = self.data_dim
        else:
            inp_dim = self.layer_dims[layer_idx - 1]
        op_dim = self.layer_dims[layer_idx]
        self.module_encoder.append(
            nn.Sequential(
                nn.Linear(inp_dim, op_dim),
                get_Activation(self.layer_activation)
            )
        )
        # Swap the values  for decoder
        inp_dim, op_dim = op_dim, inp_dim

        # Last layer
        if layer_idx == 0:
            act = self.op_activation
        else:
            act = self.layer_activation
        # Insert at start
        self.module_decoder.insert(
            0,
            nn.Sequential(
                nn.Linear(inp_dim, op_dim),
                get_Activation(act)
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

        if self.mode == 'dual':
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


class DCN():

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
            k=3,
            Lambda=0.1,
            stop_threshold=0.05,
            checkpoint_dir=None
    ):
        self.log_interval = 100
        self.device = device
        self.num_clusters = k
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
        self.Lambda = Lambda
        self.stop_threshold = stop_threshold
        self.max_loss_dec_epochs = 5
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
        print('greedy layerwise pretarining started')
        self.ae.mode = 'ae'
        self.ae.train()

        num_epochs = self.num_epochs_1
        batch_size = self.batch_size
        log_interval = 1500
        print(' Num AE layers ', self.ae_num_layers)

        for l_idx in range(self.ae_num_layers):
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
                    x_R = self.ae(x)
                    b_loss = F.mse_loss(
                        input=x, target=x_R
                    )
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
                x_R = self.ae(x)
                b_loss = F.mse_loss(
                    input=x, target=x_R
                )
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

        # =========================

    # Obtain the centroids
    # =========================
    def init_centroids(self, data):
        print("In init_centroids")
        self.ae.mode = 'dual'
        
        batch_size = self.batch_size
        z = []
        
        num_batches = data.shape[0] // batch_size + 1
        for b in range(num_batches):
            _x = data[b * batch_size: (b + 1) * batch_size]
            _x = FT(_x).to(self.device)
            _z, _ = self.ae(_x)
            z.extend(_z.cpu().data.numpy())
        z = np.array(z)

        kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            random_state=0,
            batch_size=batch_size,
            max_iter=100
        ).fit(z)

        centroids = kmeans.cluster_centers_
        print(centroids)
        print('Exiting init_centroids')
        
        return centroids

    # =========================
    # Input : embedding of data point
    # Calculate distance of each point from centroid
    # =========================
    def calculate_centroid_distance(self, z):

        z1 = z.repeat(1, self.num_clusters).reshape([-1, self.num_clusters, self.latent_dim])
        # Euclidean distance
        _centroids = self.centroids
        dist = torch.sqrt(
            torch.sum(
                (z1 - _centroids) ** 2,
                dim=-1,
                keepdim=False
            )
        )
        return dist

    # ------------
    # This should return a matrix of shape [batch, num_cluster]
    # There should be a single 1 in each row
    # ------------
    def get_cluster_assignments(self, z):

        _dist_ = self.calculate_centroid_distance(z)
        c_idx = torch.min(
            _dist_,
            dim=1,
            keepdim=False
        )[1]
        c_idx = c_idx.long()
        C = F.one_hot(c_idx, num_classes=self.num_clusters)
        return C

    # ------------------
    # Clustering loss
    # cluster_assignments : One hot vector per sample
    # -------------------
    def calc_clus_loss(self, cluster_assignments, z):
        
        _centroids = self.centroids
        P = _centroids[
            torch.max(cluster_assignments, dim=1, keepdim=False)[1]
        ]    
    
        Q = z
        dist = torch.sum((P - Q) ** 2, dim=1, keepdim=False)
        _loss = torch.sum(dist, dim=0, keepdim=False)
        
        return _loss

    # -----------------
    # # M_k = M_k - (1/ c_k )( M_k - z)
    # -----------------
    def update_cluster_centroids(self, z, C):

        counts = torch.sum(C, dim=0, keepdim=False) + 1
        _centroids = self.centroids

        z1 = z.repeat(1, self.num_clusters).reshape([-1, self.num_clusters, self.latent_dim])
        z2 = (_centroids - z1)  # distance
        # Mask
        mask = C.float()
        mask = mask.repeat(
            1, self.latent_dim
        ).reshape(
            [-1, self.latent_dim, self.num_clusters, ]
        ).permute([0, 2, 1])
        E = mask * z2

        # Sum them along axis for each cluster
        E1 = torch.sum(E, dim=0, keepdim=False)
        denom = torch.reciprocal(counts.float()).unsqueeze(1)
        E2 = denom * E1
        self.centroids = _centroids - E2

        return

        # =====================================

    # main training function
    # =====================================
    def train_model(self, data):
        ae_weights_file = os.path.join(self.checkpoint_dir, 'ae_model.pt')
        if os.path.exists(ae_weights_file) and False:
            if self.FLAG_ae_setup is False:
                # Set up the structure first , then load the weights
                for l_idx in range(self.ae_num_layers):
                    self.ae.add_layer(l_idx)
                
                checkpoint = torch.load(ae_weights_file)
                self.ae.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.pretrain_ae(data)
            # torch.save(self.ae.state_dict(), ae_weights_file)
            torch.save({
                'model_state_dict': self.ae.state_dict()
            }, ae_weights_file)
            self.FLAG_ae_setup = True
            
        self.ae = self.ae.to(self.device)
        # =======================
        self.ae.train()
        self.ae.mode = 'dual'
        
        num_epochs = self.num_epochs_2
        batch_size = self.batch_size
        num_batches = data.shape[0] // batch_size + 1
        _params = list(self.ae.parameters())

        opt = torch.optim.Adam(
            _params,
            lr=self.LR
        )
        print('Optimizer ', opt)

        # ------------
        # Initial cluster centroids
        # ------------
        self.centroids = self.init_centroids(data)
        self.centroids = FT(self.centroids).to(self.device)

#         print('Initial centroids ', self.centroids)
        print('<------------->')
        prev_epoch_loss_mean = 0

        for e in tqdm(range(1, num_epochs + 1)):
            print('Epoch :: {}'.format(e))
            np.random.shuffle(data)
            # ------------
            # cluster assignment s_ij
            # ------------
            # cluster_assignments = self.get_cluster_assignments(data)

            self.ae.mode = 'dual'
            epoch_loss = []
            
            for b in range(num_batches):
                # ==================
                # Step 1
                # ==================
            
                _x = data [b * batch_size : (b + 1) * batch_size]
                _x = torch.FloatTensor(_x)
                
                _x = _x.to(self.device)
                opt.zero_grad()
                
                z, x_r = self.ae(_x)
                
                b_cluster_assignments = self.get_cluster_assignments(z)
                
                # Calculate the reconstruction loss
                
                ae_loss = F.mse_loss(_x, x_r, reduction='none')              
                ae_loss = torch.sum(ae_loss, dim=-1, keepdim=False)
                ae_loss = torch.mean(ae_loss, dim=0, keepdim=False)
                
                # ------------
                # b_cluster_assignments = cluster_assignments[b * batch_size: (b + 1) * batch_size]
                # ------------
                # Calculate the clustering loss
                clustering_loss = self.calc_clus_loss(
                    b_cluster_assignments,
                    z
                )
                
                loss = ae_loss + self.Lambda * clustering_loss
                loss.backward(retain_graph=True)
                epoch_loss.append(loss.cpu().data.numpy())
                opt.step()
                
                if b % self.log_interval == 0:
                    print('Loss {:4f}'.format(loss.cpu().data.numpy()))
                self.ae.eval()

                # ==================
                # Step 2 
                # Calculate new assignments for batch samples
                # ==================
                z, _ = self.ae(_x)
                b_cluster_assignments = self.get_cluster_assignments(z)

                # ====================
                # Step 3
                # Update the cluster centroids

                self.update_cluster_centroids(
                    z, b_cluster_assignments
                )

                # print(self.centroids)
            epoch_loss_mean = np.mean(epoch_loss)
            diff = abs(epoch_loss_mean - prev_epoch_loss_mean)
            if diff < self.break_threshold and e > self.min_epochs:
                print('Breaking training loss staying same ')

            prev_epoch_loss_mean = epoch_loss_mean
        print('Final centroids ', self.centroids)

        # -------------
        # Save model
        # -------------



        return

    def get_cluster(self, data):
        batch_size = self.batch_size
        num_batches = data.shape[0] // batch_size + 1
        C = []
        for b in range(num_batches):
            _x = data[b * batch_size: (b + 1) * batch_size]
            _x = FT(_x).to(self.device)
            z = self.ae(_x)
            _q_ = self.calc_q_ij(z)
            _c_ = torch.max(
                _q_,
                dim=1,
                keepdim=False
            )[1]

            C.append(_c_)
        C = torch.cat(C, dim=0)
        return C.cpu().data.numpy()

    # ========================
    # Score a single sample
    # ========================
    def __score_sample(self, x):
        self.ae.eval()
        self.ae.mode = 'encoder'
        z = self.ae(x)
        cluster_idx = torch.max(self.get_cluster_assignments(z), dim=1)[1]
        c = self.centroids[cluster_idx]
        D = torch.sum((c - z) ** 2, dim=1, keepdim=False)
        return D

    def score_samples(self, data):
        bs = self.batch_size
        num_batches = data.shape[0] // bs + 1
        res = []

        for b in tqdm(range(num_batches)):
            x = data[b * bs: (b + 1)* bs]
            x = FT(x).to(self.device)
            if x.shape[0] == 0 :
                break
            r = self.__score_sample(x)
            r = r.cpu().data.numpy()
            res.extend(r)
        res = np.array(res)
        return res


'''
  def get_cluster_assignments(self, z):
        batch_size = self.batch_size
        num_batches = data.shape[0] // batch_size + 1
        self.ae.mode = 'dual'
        C = []
        for b in range(num_batches):
            _x = data[b * batch_size: (b + 1) * batch_size]
            _x = FT(_x).to(self.device)
            
            z, _  = self.ae(_x) # ae.mode is "dual" 
            _dist_ = self.calc_centroid_distance(z)
          
            c_idx = torch.min(
                    _dist_,
                    dim=1,
                    keepdim=False
                )[1]
            c_idx = c_idx.long() 
            c_idx= F.one_hot(c_idx, num_classes=self.num_clusters)
            
            C.append(c_idx)       
        C = torch.cat(C,dim=0)
        return C

'''
