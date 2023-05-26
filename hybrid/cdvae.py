import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from config import cfg
from gnn_2d import Encoder as source_encoder
from gnn_3d import Encoder as target_encoder
import numpy as np
from gemnet.gemnet import GemNetT

class crysDVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = source_encoder()
        self.target_encoder = target_encoder()
        self.decoder = GemNetT(
            num_targets=1,
            latent_dim=cfg.GNN.hidden_dim,
            emb_size_atom=cfg.GNN.hidden_dim,
            emb_size_edge=cfg.GNN.hidden_dim,
            regress_forces=True,
            cutoff=cfg.cutoff,
            max_neighbors=cfg.max_neighbors,
            otf_graph=True,
            scale_file='./gemnet/gemnet-dT.json',
        )
        self.mu_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.sigma_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.lattice_fc = nn.Linear(cfg.GNN.hidden_dim, 6)
        self.lattice_scaler = torch.load(cfg.data_dir + cfg.dataset + "_LATTICE-SCALER.pt")
        self.atom_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atomic_num)
        self.num_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atoms + 1)
        self.proj = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))

        self.cos = nn.CosineSimilarity(dim=1)
        self.ce = nn.CrossEntropyLoss()
        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

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

    def forward(self, gg, fg, gprop, yprop):
        z1 = self.source_encoder(fg)
        zn, z2 = self.target_encoder(gg)
        mu, logvar = self.mu_fc(z2), self.sigma_fc(z2)
        z2 = self.reparameterize(mu, logvar)
        cos_loss = -(self.cos(self.proj(z1), z2).mean())

        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gprop)
        latt_loss *= 10

        pred_comp_per_atom = self.predict_atom(z2, gprop.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gprop)

        pred_nums = self.predict_nums(z2)
        num_loss = self.num_loss(pred_nums, gprop)

        kld_loss = self.kld_loss(mu, logvar)
        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (gprop.num_atoms.size(0),),
                                    device=self.device)

        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            gprop.num_atoms, dim=0)

        cart_noises_per_atom = (
                torch.randn_like(gprop.frac_coords) *
                used_sigmas_per_atom[:, None])

        cart_coords = gprop.frac_coords + cart_noises_per_atom
        pred_cart_coord_diff, pred_atom_types = self.decoder(
            z2, cart_coords, gprop.atom_types, gprop.num_atoms, gprop.lenghts, gprop.angles)

        '''
        
        noise = torch.randn_like(nX3, with noise level)
        pred_noise = self.decoder_gemnet(z2, noise, atom_type, real_len, real_angle)
        coord_loss = self.coord_loss(noise, pred_noise)
        
        evaluate
        z1 = source_enc(x)
        pred_noise = self.decoder(z1, random_noise)
        coord = random_noise - pred_noise
        test_loss = (real_coord, coord)
        or
        make graph with fg x and coord, train alignn, compare with real coord train alignn
        
        '''


        loss = cycle_loss + atom_loss + latt_loss + kld_loss  + num_loss#+ nce_loss
        loss_dict = {
            # 'cos_loss': cos_loss,
            'cycle_loss': cycle_loss,
            'atom_loss': atom_loss,
            'latt_loss': latt_loss,
            'kld_loss': kld_loss,
            'num_loss': num_loss
            # 'nce_loss':nce_loss,
        }
        return loss, loss_dict

    def predice_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_length_and_angles = self.lattice_fc(z)
        scaled_preds = self.lattice_scaler.inverse_transform(pred_length_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
        # w/o lattice scaling
        # self.lattice_scaler.match_device(z)
        # pred_length_and_angles = self.lattice_fc(z)
        # pred_lengths = pred_length_and_angles[:, :3]
        # pred_angles = pred_length_and_angles[:, 3:]
        # pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)

        return pred_length_and_angles, pred_lengths, pred_angles

    def lattice_loss(self, pred_lengths_and_angles, gdata):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        target_lengths_and_angles = self.lattice_scaler.transform(gdata.lscaled_lattice)
        # w/o lattice scaling
        # target_lengths_and_angles = gdata.lscaled_lattice
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def predict_atom(self, z, num_atoms):
        # with readout
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_comp_per_atom = self.atom_fc(z_per_atom)
        # with node represen
        # pred_comp_per_atom = self.atom_fc(z)
        return pred_comp_per_atom

    def predict_nums(self, z):
        return self.num_fc(z)

    def atom_loss(self, pred_comp_per_atom, gdata):
        target_atomic_num = gdata.atomic_nums -1
        loss = F.cross_entropy(pred_comp_per_atom, target_atomic_num, reduction='none')
        return scatter(loss, gdata.batch, reduce='mean').mean()

    def num_loss(self, pred_num_atoms, gdata):
        return F.cross_entropy(pred_num_atoms, gdata.num_atoms)

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss
