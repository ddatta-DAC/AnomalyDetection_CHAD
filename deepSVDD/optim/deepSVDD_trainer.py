import logging
import time
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch import FloatTensor as FT
from torch import LongTensor as LT
from torch import nn

class DeepSVDDTrainer():
    def __init__(
            self,
            DEVICE,
            objective,
            R,
            c,
            nu,
            LR=0.001,
            num_epochs=50,
            batch_size=256,
            warm_up_epochs = 10,
            ae_epochs = 10
    ):
        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        self.device = DEVICE
        self.ae_epochs = ae_epochs
        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        # Optimization parameters
        # number of training epochs for soft-boundary Deep SVDD before radius R gets updated
        self.warm_up_n_epochs = warm_up_epochs
        self.LR = LR
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.net = None
        
        return

    def train(self, net, net_i, train_X):
        # Set device for network
        net = net.to(self.device)
        net_i = net_i.to(self.device)
        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(
            net.parameters(),
            lr=self.LR
        )

        
        net.train()
        net_i.train()
        print(' AE epochs >>',self.ae_epochs)
        opt_ae = optim.Adam(
            list(net.parameters()) + list(net_i.parameters()),
            lr=self.LR*2
        )

        # ---- Do simple AE training --- #
        num_batches = train_X.shape[0] // self.batch_size
        for epoch in tqdm(range(self.ae_epochs)):
            np.random.shuffle(train_X)
            loss_vals = []
            for b in range(num_batches):
                opt_ae.zero_grad()

                _x0 = FT(train_X[b * self.batch_size:(b + 1) * self.batch_size]).to(self.device)
                _x = net(_x0)
                _x1 = net_i(_x)
                _loss = F.mse_loss(
                    _x0,_x1,reduction='none'
                )
                _loss = torch.sum(_loss, dim=-1, keepdim=False)
                _loss = torch.mean(_loss,dim=0)
                _loss.backward()
                opt_ae.step()
                loss_vals.append(_loss.cpu().data.numpy())
            print('AE training Epoch {} Loss {:4f}'.format(epoch, np.mean(loss_vals)))
        print('AE training done')

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(train_X, net)
            print('Center c initialized.')

        num_batches = train_X.shape[0] // self.batch_size
        
        for epoch in tqdm(range(self.num_epochs)):
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            np.random.shuffle(train_X)
            for b in range(num_batches):
                inputs = train_X[b * self.batch_size:(b + 1) * self.batch_size]
                inputs = FT(inputs).to(self.device)
                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(
                epoch + 1, self.num_epochs, epoch_train_time, loss_epoch / num_batches)
            )

        print('Finished training.')
        self.net = net
        return net

    def test(self, test_X):
        logger = logging.getLogger()
        # Set device for network
        batch_size = 517
        num_batches = test_X.shape[0] // batch_size + 1

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        self.net.eval()

        scores_list = []
        with torch.no_grad():
            for b in range(num_batches):
                _x = test_X[b * batch_size:(b + 1) * batch_size]
                _x = FT(_x).to(self.device)
                outputs = self.net(_x)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist
                scores_list.extend(scores.cpu().data.numpy().tolist())
                # Save triples of (idx, label, score) in a list
                # idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                #                             labels.cpu().data.numpy().tolist(),
                #                             scores.cpu().data.numpy().tolist()))

        return np.array(scores_list)
        # self.test_time = time.time() - start_time
        # logger.info('Testing time: %.3f' % self.test_time)

        # self.test_scores = idx_label_score
        #
        # # Compute AUC
        # _, labels, scores = zip(*idx_label_score)
        # labels = np.array(labels)
        # scores = np.array(scores)
        #
        # self.test_auc = roc_auc_score(labels, scores)
        # logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
        #
        # logger.info('Finished testing.')
    def init_center_c(self, x, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)
        num_batches = x.shape[0] // self.batch_size + 1
        bs = self.batch_size

        net.eval()
        with torch.no_grad():
            for b in range(num_batches):
                # get the inputs of the batch
                inputs = x[b * bs:(b + 1) * bs]
                inputs = FT(inputs).to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
