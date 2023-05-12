import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, global_add_pool
from config import cfg

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = cfg.FORMULA.hidden_dim
        self.atom_emb = nn.Sequential(nn.Linear(cfg.FORMULA.atom_input_dim, self.hidden_dim),
                                      nn.BatchNorm1d(self.hidden_dim),
                                      nn.ReLU())
        self.enc = GINConv(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                         nn.BatchNorm1d(self.hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(self.hidden_dim, self.hidden_dim)))
        self.enc_modules = nn.ModuleList([self.enc for _ in range(cfg.FORMULA.layers)])
        self.fc_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_var = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dec = nn.Sequential(GATConv(self.hidden_dim, self.hidden_dim, heads = cfg.FORMULA.n_heads),
                                 nn.Linear(cfg.FORMULA.n_heads*self.hidden_dim, self.hidden_dim))
        self.dec_modules = nn.ModuleList([self.dec for _ in range(cfg.FORMULA.layers)])

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, batch):
        x = batch.node_features_s
        e = batch.edge_index_s
        w = batch.atom_weights_s
        x = self.atom_emb(x)
        for module in self.enc_modules:
            x = module(x, e)
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        z = self.reparameterize(mu, logvar)
        zs = (z, z)
        zin, zout = zs[0], zs[1]
        for module in self.dec_modules:
            zout = module[0](zout,e)
            zout = module[1](zout)
        return zin, zout, mu, logvar




