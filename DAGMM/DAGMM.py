import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import itertools
from torch import FloatTensor as FT

try:
    from utils import *
except:
    from .utils import *


# class Cholesky(torch.autograd.Function):
#     def forward(ctx, a):
#         l = torch.cholesky(a, False)
#         ctx.save_for_backward(l)
#         return l
#
#     def backward(ctx, grad_output):
#         l, = ctx.saved_variables
#         linv = l.inverse()
#         inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
#             1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
#         s = torch.mm(linv.t(), torch.mm(inner, linv))
#         return s


class AE_encoder(nn.Module):
    def __init__(
            self,
            device,
            structure_config
    ):

        super(AE_encoder, self).__init__()
        self.device = device
        self.structure_config = structure_config
        activation = structure_config['encoder_layers']['activation']
        layer_dims = structure_config['encoder_layers']['layer_dims']

        # ================
        # Concatenate the input
        # ================

        self.discrete_column_dims = structure_config['discrete_column_dims']
        self.num_discrete_columns = structure_config['num_discrete']
        self.num_real_columns = structure_config['num_real']

        input_dim = self.num_discrete_columns + self.num_real_columns
        layers = []
        # Pairs <  output_dim, activation >
        num_layers = len(layer_dims)
        for idx in range(num_layers):
            op_dim = layer_dims[idx]
            layers.append(nn.Linear(
                input_dim, op_dim
            ))
            if idx == num_layers - 1: activation = 'none'
            if activation == 'tanh':
                layers.append(
                    nn.Tanh()
                )
            elif activation == 'relu':
                layers.append(
                    nn.ReLU()
                )
            elif activation == 'sigmoid':
                layers.append(
                    nn.Sigmoid()
                )
            elif activation == 'none':
                pass

            input_dim = op_dim

        self.FC_z = nn.Sequential(*layers)
        return

    def forward(self, X):

        # real_x_0 = X[:, -self.num_real_columns:].type(torch.FloatTensor).to(self.device)
        # discrete_x_0 = X[:, :self.num_discrete_columns].type(torch.LongTensor).to(self.device)
        # discrete_x_1 = torch.chunk(discrete_x_0, self.num_discrete_columns, dim=1)
        # # discrete_x_1 is an array
        # res = []
        # column_name_list = list(self.discrete_column_dims.keys())
        # for idx in range(self.num_discrete_columns):
        #     col = column_name_list[idx]
        #     _x = discrete_x_1[idx].to(self.device)
        #     n_cat = self.discrete_column_dims[col]
        #     _x = F.one_hot(_x, n_cat).type(FT).squeeze(1).to(self.device)
        #     res.append(_x)
        #
        # res.append(real_x_0)
        # x_concat = torch.cat(res, dim=1)

        # ==============
        # Conactenated input : res
        # ==============
        op = self.FC_z(X)
        return op


class AE_decoder(nn.Module):
    def __init__(
            self,
            device,
            structure_config=None,
    ):
        super(AE_decoder, self).__init__()
        self.device = device
        self.structure_config = structure_config
        self.discrete_column_dims = structure_config['discrete_column_dims']
        self.num_discrete_columns = structure_config['num_discrete']
        self.num_real_columns = structure_config['num_real']
        final_output_dim = structure_config['final_output_dim']
        activation = structure_config['decoder_layers']['activation']
        layer_dims = structure_config['decoder_layers']['layer_dims']

        layers = []
        inp_dim = layer_dims[0]
        num_layers = len(layer_dims)
        for idx in range(1, num_layers):
            op_dim = layer_dims[idx]
            layers.append(nn.Linear(inp_dim, op_dim))
            if activation == 'tanh':
                layers.append(
                    nn.Tanh()
                )
            elif activation == 'sigmoid':
                layers.append(
                    nn.Sigmoid()
                )
            elif activation == 'relu':
                layers.append(
                    nn.ReLU()
                )
            elif activation == 'none':
                layers.append(
                    nn.Sigmoid()
                )
            inp_dim = op_dim

        self.FC_z = nn.Sequential(*layers)
        return

    def forward(self, z):
        res = self.FC_z(z)
        return res


class AE(nn.Module):

    def __init__(
            self,
            device,
            encoder_structure_config,
            decoder_structure_config
    ):
        super(AE, self).__init__()
        self.device = device
        self.encoder = AE_encoder(
            device,
            encoder_structure_config
        )
        self.decoder = AE_decoder(
            device,
            decoder_structure_config
        )

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.mode = None
        return

    def forward(self, x):
        z = self.encoder(x)
        if self.mode == 'compress':
            return z
        x_recon = self.decoder(z)
        return x_recon, z


class DaGMM(nn.Module):

    def __init__(
            self,
            device,
            encoder_structure_config,
            decoder_structure_config,
            n_gmm=2,
            ae_latent_dim=1,
            fc_dropout=0.5
    ):

        super(DaGMM, self).__init__()
        self.device = device
        self.fc_dropout = fc_dropout
        self.encoder = AE_encoder(
            device,
            encoder_structure_config
        )
        self.encoder = self.encoder.to(self.device)
        self.decoder = AE_decoder(
            device,
            decoder_structure_config
        )

        self.decoder = self.decoder.to(self.device)
        latent_dim = ae_latent_dim + 2
        layers = []
        layers += [nn.Linear(latent_dim, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=self.fc_dropout)]
        layers += [nn.Linear(10, n_gmm)]
        layers += [nn.Softmax(dim=1)]

        self.estimation = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, latent_dim, latent_dim))

    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(x, dec)
        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
        gamma = self.estimation(z)
        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D, _ = cov.size()
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D) * eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            tmp = np.linalg.det(cov_k.data.cpu().numpy() * (2 * np.pi))
            det_cov.append(tmp)
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(
            torch.sum(
                z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2
            ) * z_mu,
            dim=-1
        )
        # for stability (logsumexp)
        k = (exp_term_tmp).clamp(min=0)

        max_val = torch.max(k, dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = torch.mean((x - x_hat) ** 2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag