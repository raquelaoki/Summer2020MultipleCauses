# Reference https://github.com/kim-hyunsu/CEVAE-pyro/blob/master/model/vae.py
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.distributions
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

print('Available devices ', torch.cuda.device_count())
print('Current cuda device ', torch.cuda.current_device())


class Data(object):
    # replications were over the treatments
    def __init__(self, X_train, X_test, y_train, y_test, treatments_columns, data_path, binfeats=None, contfeats=None):
        self.treatments_columns = treatments_columns
        self.data_path = data_path
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.binfeats = list(range(self.X.shape[1])) if binfeats is None else binfeats  # which features are continuous
        self.contfeats = [] if contfeats is None else contfeats  # which features are continuous
        # TODO: update continuous features for goPDX

    def get_train_valid_test(self):
        for col in self.treatments_columns:
            dataset_train = TensorDataset(Tensor(np.delete(self.X_train, col, 1)), Tensor(self.X_train[:, col]),
                                          Tensor(self.y_train))
            dataset_test = TensorDataset(Tensor(np.delete(self.X_test, col, 1)), Tensor(self.X_test[:, col]),
                                         Tensor(self.y_test))

            ''' Required: Create DataLoader for training the models '''
            loader_train = DataLoader(dataset_train, shuffle=True, batch_size=params["batch_size"])
            loader_test = DataLoader(dataset_test, shuffle=False, batch_size=len(rows_test))

            yield loader_train, loader_test, self.contfeats, self.binfeats


def get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, x_train, t_train, L=1):
    y_infer = q_y_xt_dist(x_train.float(), t_train.float())
    # use inferred y
    xy = torch.cat((x_train.float(), y_infer.mean), 1)  # TODO take mean?
    z_infer = q_z_tyx_dist(xy=xy, t=t_train.float())
    # Manually input zeros and ones
    y0 = p_y_zt_dist(z_infer.mean, torch.zeros(z_infer.mean.shape).cuda()).mean  # TODO take mean?
    y1 = p_y_zt_dist(z_infer.mean, torch.ones(z_infer.mean.shape).cuda()).mean  # TODO take mean?

    return y0.cpu().detach().numpy(), y1.cpu().detach().numpy()


def init_qz(qz, pz, data_loader):
    """
    Initialize qz towards outputting standard normal distributions
    - with standard torch init of weights the gradients tend to explode after first update step
    """
    batch = next(iter(data_loader))
    optimizer = optim.Adam(qz.parameters(), lr=0.001)

    for i in range(50):
        xy = torch.cat((batch[0], batch[2]), 1)
        z_infer = qz(xy=xy, t=batch[1])
        # KL(q_z|p_z) mean approx, to be minimized
        # KLqp = (z_infer.log_prob(z_infer.mean) - pz.log_prob(z_infer.mean)).sum(1)
        # Analytic KL
        KLqp = (-torch.log(z_infer.stddev) + 1 / 2 * (z_infer.variance + z_infer.mean ** 2 - 1)).sum(1)

        objective = KLqp
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        if KLqp != KLqp:
            raise ValueError('KL(pz,qz) contains NaN during init')
    return qz


class p_x_z(nn.Module):

    def __init__(self, dim_in=20, nh=3, dim_h=20, dim_out_bin=19, dim_out_con=6):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out_bin = dim_out_bin
        self.dim_out_con = dim_out_con

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh - 1)])
        # output layer defined separate for continuous and binary outputs
        self.output_bin = nn.Linear(dim_h, dim_out_bin)
        # for each output an mu and sigma are estimated
        self.output_con_mu = nn.Linear(dim_h, dim_out_con)
        self.output_con_sigma = nn.Linear(dim_h, dim_out_con)
        self.softplus = nn.Softplus()

    def forward(self, z_input):
        z = F.elu(self.input(z_input))
        for i in range(self.nh - 1):
            z = F.elu(self.hidden[i](z))
        # for binary outputs:
        x_bin_p = torch.sigmoid(self.output_bin(z))
        x_bin = bernoulli.Bernoulli(x_bin_p)
        # for continuous outputs
        mu, sigma = self.output_con_mu(z), self.softplus(self.output_con_sigma(z))
        x_con = normal.Normal(mu, sigma)

        if (z != z).all():
            raise ValueError('p(x|z) forward contains NaN')

        return x_bin, x_con


class p_t_z(nn.Module):

    def __init__(self, dim_in=20, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # for binary outputs:
        out_p = torch.sigmoid(self.output(x))

        out = bernoulli.Bernoulli(out_p)
        return out


class p_y_zt(nn.Module):

    def __init__(self, dim_in=20, nh=3, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # Separated forwards for different t values, TAR

        self.input_t0 = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden_t0 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.mu_t0 = nn.Linear(dim_h, dim_out)

        self.input_t1 = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden_t1 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.mu_t1 = nn.Linear(dim_h, dim_out)

    def forward(self, z, t):
        # Separated forwards for different t values, TAR

        x_t0 = F.elu(self.input_t0(z))
        for i in range(self.nh):
            x_t0 = F.elu(self.hidden_t0[i](x_t0))
        mu_t0 = F.elu(self.mu_t0(x_t0))

        x_t1 = F.elu(self.input_t1(z))
        for i in range(self.nh):
            x_t1 = F.elu(self.hidden_t1[i](x_t1))
        mu_t1 = F.elu(self.mu_t1(x_t1))
        # set mu according to t value
        y = normal.Normal((1 - t) * mu_t0 + t * mu_t1, 1)

        return y


class q_t_x(nn.Module):

    def __init__(self, dim_in=25, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # for binary outputs:
        out_p = torch.sigmoid(self.output(x))
        out = bernoulli.Bernoulli(out_p)

        return out


class q_y_xt(nn.Module):

    def __init__(self, dim_in=25, nh=3, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        # separate outputs for different values of t
        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)

    def forward(self, x, t):
        # Unlike model network, shared parameters with separated heads
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # only output weights separated
        mu_t0 = self.mu_t0(x)
        mu_t1 = self.mu_t1(x)
        # set mu according to t, sigma set to 1
        y = normal.Normal((1 - t) * mu_t0 + t * mu_t1, 1)
        return y


class q_z_tyx(nn.Module):

    def __init__(self, dim_in=25 + 1, nh=3, dim_h=20, dim_out=20):
        super().__init__()
        # dim in is dim of x + dim of y
        # dim_out is dim of latent space z
        # save required vars
        self.nh = nh

        # Shared layers with separated output layers

        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])

        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)
        self.sigma_t0 = nn.Linear(dim_h, dim_out)
        self.sigma_t1 = nn.Linear(dim_h, dim_out)
        self.softplus = nn.Softplus()

    def forward(self, xy, t):
        # Shared layers with separated output layers
        # print('before first linear z_infer')
        # print(xy)
        x = F.elu(self.input(xy))
        # print('first linear z_infer')
        # print(x)
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))

        mu_t0 = self.mu_t0(x)
        mu_t1 = self.mu_t1(x)
        sigma_t0 = self.softplus(self.sigma_t0(x))
        sigma_t1 = self.softplus(self.sigma_t1(x))

        # Set mu and sigma according to t
        z = normal.Normal((1 - t) * mu_t0 + t * mu_t1, (1 - t) * sigma_t0 + t * sigma_t1)
        return z


class CEVAE():
    def __init__(self, X_train, X_test, y_train, y_test,
                 treatments_columns, data_path, z_dim=20,
                 h_dim=64, epochs=100, batch=500, lr=0.001,
                 decay=0.001, print_every=100):
        super(CEVAE, self).__init__()
        self.treatments_columns = treatments_columns
        self.dataset = Data(X_train, X_test, y_train, y_test, treatments_columns, data_path)
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.decay = decay
        self.print_every = print_every
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # TODO: Here

    def fit_all(self):
        cevae_cate = np.zeros(len(self.treatments_columns))
        for i, (train_loader, test_loader, contfeats, binfeats) in enumerate(self.dataset.get_train_valid_test()):
            # train contains: X, t, y
            y0, y1, cevae_cate[i], y_test_pred, y_test = self.fit(train_loader, test_loader)
        return cevae_cate

    def fit(self, train_loader, test_loader):
        # read out data
        # (xtr, ttr, ytr) = train
        # (xva, tva, yva) = valid  # not being used
        # (xte, tte, yte) = test

        try:
            # init networks (overwritten per replication)
            x_dim = len(self.dataset.binfeats) + len(self.dataset.contfeats)
            p_x_z_dist = p_x_z(dim_in=self.z_dim, nh=3, dim_h=self.h_dim, dim_out_bin=len(self.dataset.binfeats),
                               dim_out_con=len(self.dataset.contfeats)).cuda()
            p_t_z_dist = p_t_z(dim_in=self.z_dim, nh=1, dim_h=self.h_dim, dim_out=1).cuda()
            p_y_zt_dist = p_y_zt(dim_in=self.z_dim, nh=3, dim_h=self.h_dim, dim_out=1).cuda()
            q_t_x_dist = q_t_x(dim_in=x_dim, nh=1, dim_h=self.h_dim, dim_out=1).cuda()
            # t is not feed into network, therefore not increasing input size (y is fed).
            q_y_xt_dist = q_y_xt(dim_in=x_dim, nh=3, dim_h=args.h_dim, dim_out=1).cuda()
            q_z_tyx_dist = q_z_tyx(dim_in=len(self.dataset.binfeats) + len(self.dataset.contfeats) + 1, nh=3,
                                   dim_h=self.h_dim,
                                   dim_out=self.z_dim).cuda()  # remove an 1 from dim_in
            p_z_dist = normal.Normal(torch.zeros(self.z_dim).cuda(), torch.ones(self.z_dim).cuda())
            # Create optimizer
            model = list(p_x_z_dist.parameters()) + list(p_t_z_dist.parameters()) + \
                    list(p_y_zt_dist.parameters()) + list(q_t_x_dist.parameters()) + \
                    list(q_y_xt_dist.parameters()) + list(q_z_tyx_dist.parameters())
            # Adam is used, like original implementation, in paper Adamax is suggested
            optimizer = optim.Adam(model, lr=self.lr, weight_decay=self.decay)
            # init q_z inference
            q_z_tyx_dist = init_qz(q_z_tyx_dist, p_z_dist, train_loader)
            loss = defaultdict(list)
            for epoch in range(self.epochs):
                # batch: X, t, y
                for i, batch in enumerate(tqdm(train_loader)):
                    # inferred distribution over z
                    xy = torch.cat((batch[0], batch[2]), 1)
                    z_infer = q_z_tyx_dist(xy=xy, t=batch[1])
                    # use a single sample to approximate expectation in lowerbound
                    z_infer_sample = z_infer.sample()

                    # RECONSTRUCTION LOSS
                    # p(x|z)
                    x_bin, x_con = p_x_z_dist(z_infer_sample)
                    l1 = x_bin.log_prob(atch[0][:, :len(self.dataset.binfeats)]).sum(1)
                    loss['Reconstr_x_bin'].append(l1.sum().cpu().detach().float())
                    # l2 = x_con.log_prob(x_train[:, -len(contfeats):]).sum(1)
                    # loss['Reconstr_x_con'].append(l2.sum().cpu().detach().float())
                    # p(t|z)
                    t = p_t_z_dist(z_infer_sample)
                    l3 = t.log_prob(batch[1]).squeeze()
                    loss['Reconstr_t'].append(l3.sum().cpu().detach().float())
                    # p(y|t,z)
                    # for training use t_train, in out-of-sample prediction this becomes t_infer
                    y = p_y_zt_dist(z_infer_sample, batch[1])
                    l4 = y.log_prob(y_train).squeeze()
                    loss['Reconstr_y'].append(l4.sum().cpu().detach().float())

                    # REGULARIZATION LOSS
                    # p(z) - q(z|x,t,y)
                    # approximate KL
                    l5 = (p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(1)
                    # Analytic KL (seems to make overall performance less stable)
                    # l5 = (-torch.log(z_infer.stddev) + 1/2*(z_infer.variance + z_infer.mean**2 - 1)).sum(1)
                    loss['Regularization'].append(l5.sum().cpu().detach().float())

                    # AUXILIARY LOSS
                    # q(t|x)
                    t_infer = q_t_x_dist(batch[0])
                    l6 = t_infer.log_prob(batch[1]).squeeze()
                    loss['Auxiliary_t'].append(l6.sum().cpu().detach().float())
                    # q(y|x,t)
                    y_infer = q_y_xt_dist(batch[0], batch[1])
                    l7 = y_infer.log_prob(batch[2]).squeeze()
                    loss['Auxiliary_y'].append(l7.sum().cpu().detach().float())

                    # Total objective
                    # inner sum to calculate loss per item, torch.mean over batch
                    loss_mean = torch.mean(l1 + l3 + l4 + l5 + l6 + l7)  # + l2
                    loss['Total'].append(loss_mean.cpu().detach().numpy())
                    objective = -loss_mean

                    optimizer.zero_grad()
                    # Calculate gradients
                    objective.backward()
                    # Update step
                    optimizer.step()

                if epoch % self.print_every == 0:
                    print('Epoch - ', epoch, ' Loss: ', loss_mean)
                    # TODO: add eval for validation and training set?
            # Done Training!
            batch = next(iter(test_loader))
            y0, y1 = get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, batch[0].cuda(), batch[1].cuda())
            y01_pred = q_y_xt_dist(batch[0].cuda(), batch[1].cuda())
            y_pred = self.scaler.fit_transform(y01_pred.mean.cpu().detach().numpy())
            # y0, y1 = y0 * ys + ym, y1 * ys + ym
            # returns fit info only for testing set
            return y0[:, 0].mean(), y1[:, 0].mean(), (y1[:, 0] - y0[:, 0]).mean(), y_pred, batch[1]
        except ValueError:
            # TODO: update except below
            y_pred = np.zeros(len(yte))
            y_pred[:] = np.nan
            y_pred = y_pred.reshape(-1, 1)
            print('ERROR:', tcol)
            return 0.0, 0.0, 0.0, y_pred, np.squeeze(np.asarray(yte))