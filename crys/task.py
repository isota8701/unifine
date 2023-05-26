import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from config import cfg
from gnn_2d import Encoder as source_encoder
from gnn_3d import Encoder as target_encoder
from gnn_3d import hybridEncoder
import numpy as np
from dgl.nn.pytorch.glob import AvgPooling



class crysDnoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = source_encoder()
        self.target_encoder = target_encoder()
        self.yprop_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.output_dim)
        self.lattice_fc = nn.Linear(cfg.GNN.hidden_dim, 6)
        self.lattice_scaler = torch.load(cfg.data_dir + cfg.dataset + "_LATTICE-SCALER.pt")
        self.atom_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atomic_num)
        self.num_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atoms+1)
        self.proj = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))
        self.invp = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))

        self.ce = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim = 1)


    def forward(self, gg, fg, gprop, yprop):
        z1, kl1= self.source_encoder(fg)
        zn, z2, kl2 = self.target_encoder(gg)

        # crys_noise_cycle
        loss_p = F.mse_loss(self.proj(z1), z2)
        loss_i = F.mse_loss(z1, self.invp(z2))
        loss_c = F.mse_loss(self.invp(self.proj(z1)), z1)
        cycle_loss = loss_p + loss_i + loss_c

        kld_loss = 0.5*(kl1+kl2)

        # crys_noise_cos
        # cos_loss = -(self.cos(self.proj(z1), z2).mean())

        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gprop)
        latt_loss*=10

        pred_comp_per_atom = self.predict_atom(zn, gprop.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gprop)

        loss =  cycle_loss + atom_loss + latt_loss + kld_loss
        loss_dict = {
            'cycle_loss': cycle_loss,
            'atom_loss': atom_loss,
            'latt_loss': latt_loss,
            'kld_loss': kld_loss,
            # 'cosine_loss': cos_loss,
            # 'y2_loss': y2_loss,
            # 'y1_loss': y1_loss
        }
        return loss, loss_dict

    def predice_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_length_and_angles = self.lattice_fc(z)
        scaled_preds = self.lattice_scaler.inverse_transform(pred_length_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:,3:]
        pred_lengths = pred_lengths*num_atoms.view(-1,1).float()**(1/3)
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
        # z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        # pred_comp_per_atom = self.atom_fc(z_per_atom)
        # with node represen
        pred_comp_per_atom = self.atom_fc(z)
        return pred_comp_per_atom

    def predict_nums(self, z):
        return self.num_fc(z)

    def atom_loss(self, pred_comp_per_atom, gdata):
        target_atomic_num = gdata.atomic_nums -1
        loss = F.cross_entropy(pred_comp_per_atom, target_atomic_num, reduction='none')
        return scatter(loss, gdata.batch, reduce='mean').mean()

    def num_loss(self, pred_num_atoms, gdata):
        return F.cross_entropy(pred_num_atoms, gdata.num_atoms)


class crysVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = source_encoder()
        self.target_encoder = target_encoder()
        self.mu_fc1 = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.sigma_fc1 = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.mu_fc2 = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.sigma_fc2 = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.yprop_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.output_dim)
        self.lattice_fc = nn.Linear(cfg.GNN.hidden_dim, 6)
        self.lattice_scaler = torch.load(cfg.data_dir + cfg.dataset + "_LATTICE-SCALER.pt")
        self.atom_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atomic_num)
        self.num_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atoms + 1)
        self.proj = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))
        self.invp = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))

        self.cos = nn.CosineSimilarity(dim=1)
        self.ce = nn.CrossEntropyLoss()

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
        zt, z2 = self.target_encoder(gg)
        # mu1, logvar1 = self.mu_fc1(z1), self.sigma_fc1(z1)
        mu2, logvar2 = self.mu_fc2(z2), self.sigma_fc2(z2)
        # z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)

        # cos_loss = -(self.cos(self.proj(z1), z2).mean())

        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gprop)
        latt_loss *= 10

        pred_comp_per_atom = self.predict_atom(zt, gprop.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gprop)



        # pred_nums = self.predict_nums(z2)
        # num_loss = self.num_loss(pred_nums, gprop)

        loss_p = F.mse_loss(self.proj(z1), z2)
        loss_i = F.mse_loss(z1, self.invp(z2))
        loss_c = F.mse_loss(self.invp(self.proj(z1)), z1)
        cycle_loss = loss_p + loss_i + loss_c

        # kld_loss1 = self.kld_loss(mu1, logvar1)
        kld_loss = self.kld_loss(mu2, logvar2)
        # kld_loss = 0.5*(kld_loss1+kld_loss2)

        # info nce loss
        # z = torch.cat([z1,z2], dim = 0)
        # z = F.normalize(z, dim = -1)
        # sim_mat = torch.matmul(z, z.T)
        #
        # lbl = torch.cat([torch.arange(z1.shape[0]) for _ in range(2)], dim = 0)
        # lbl = (lbl.unsqueeze(0) == lbl.unsqueeze(1)).float()
        # lbl = lbl.to(z.device)
        #
        # mask = torch.eye(lbl.shape[0], dtype = torch.bool).to(z.device)
        #
        # lbl = lbl[~mask].view(lbl.shape[0],-1)
        # sim_mat = sim_mat[~mask].view(sim_mat.shape[0],-1)
        #
        # positives = sim_mat[lbl.bool()].view(lbl.shape[0],-1)
        # negatives = sim_mat[~lbl.bool()].view(sim_mat.shape[0],-1)
        # logits = torch.cat([positives, negatives], dim = 1)
        # logits = logits / cfg.temperature
        # labels = torch.zeros(logits.shape[0], dtype = torch.long).to(z.device)
        # nce_loss = self.ce(logits, labels)

        loss = cycle_loss + atom_loss + latt_loss + kld_loss  #+ num_loss#+ nce_loss
        loss_dict = {
            # 'cos_loss': cos_loss,
            'cycle_loss': cycle_loss,
            'atom_loss': atom_loss,
            'latt_loss': latt_loss,
            'kld_loss': kld_loss,
            # 'num_loss': num_loss
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
        # z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        # pred_comp_per_atom = self.atom_fc(z_per_atom)
        # with node represen
        pred_comp_per_atom = self.atom_fc(z)
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


class crysNodeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = source_encoder()
        self.target_encoder = target_encoder()
        self.mu_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.sigma_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.yprop_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.output_dim)
        self.lattice_fc = nn.Linear(cfg.GNN.hidden_dim, 6)
        self.lattice_scaler = torch.load(cfg.data_dir + cfg.dataset + "_LATTICE-SCALER.pt")
        self.atom_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atomic_num)
        self.num_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atoms + 1)
        self.proj = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))
        self.invp = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))
        self.pooling = AvgPooling()
        self.cos = nn.CosineSimilarity(dim=1)
        self.ce = nn.CrossEntropyLoss()

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

    def get_indices(self, gdata, replace_ratio: float = 0.2):
        id = 0
        ids = []
        for num_atom in gdata.num_atoms:
            i = torch.randint(id, id + num_atom, (int(num_atom * replace_ratio),))
            ids.append(i)
            id += num_atom
        return torch.cat(ids)

    def forward(self, gg, fg, gprop, yprop):
        zs = self.source_encoder(fg)
        zt = self.target_encoder(gg)
        z2 = self.pooling(gg[0], zt)

        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gprop)
        latt_loss *= 10

        pred_comp_per_atom = self.predict_atom(zt, gprop.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gprop)

        zh = zs.clone()
        ind = self.get_indices(gprop, replace_ratio=0.333)
        zh[ind, :] = zt[ind, :]

        loss_p = F.mse_loss(self.proj(zh), zt)
        loss_i = F.mse_loss(zs, self.invp(zt))
        loss_c = F.mse_loss(self.invp(self.proj(zh)), zs)
        cycle_loss = loss_p + loss_i + loss_c

        # info nce loss
        # z = torch.cat([z1,z2], dim = 0)
        # z = F.normalize(z, dim = -1)
        # sim_mat = torch.matmul(z, z.T)
        #
        # lbl = torch.cat([torch.arange(z1.shape[0]) for _ in range(2)], dim = 0)
        # lbl = (lbl.unsqueeze(0) == lbl.unsqueeze(1)).float()
        # lbl = lbl.to(z.device)
        #
        # mask = torch.eye(lbl.shape[0], dtype = torch.bool).to(z.device)
        #
        # lbl = lbl[~mask].view(lbl.shape[0],-1)
        # sim_mat = sim_mat[~mask].view(sim_mat.shape[0],-1)
        #
        # positives = sim_mat[lbl.bool()].view(lbl.shape[0],-1)
        # negatives = sim_mat[~lbl.bool()].view(sim_mat.shape[0],-1)
        # logits = torch.cat([positives, negatives], dim = 1)
        # logits = logits / cfg.temperature
        # labels = torch.zeros(logits.shape[0], dtype = torch.long).to(z.device)
        # nce_loss = self.ce(logits, labels)

        loss = cycle_loss + atom_loss + latt_loss + kld_loss  # + num_loss#+ nce_loss
        loss_dict = {
            # 'cos_loss': cos_loss,
            'cycle_loss': cycle_loss,
            'atom_loss': atom_loss,
            'latt_loss': latt_loss,
            'kld_loss': kld_loss,
            # 'num_loss': num_loss
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
        # z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        # pred_comp_per_atom = self.atom_fc(z_per_atom)
        # with node represen
        pred_comp_per_atom = self.atom_fc(z)
        return pred_comp_per_atom

    def predict_nums(self, z):
        return self.num_fc(z)

    def atom_loss(self, pred_comp_per_atom, gdata):
        target_atomic_num = gdata.atomic_nums - 1
        loss = F.cross_entropy(pred_comp_per_atom, target_atomic_num, reduction='none')
        return scatter(loss, gdata.batch, reduce='mean').mean()

    def num_loss(self, pred_num_atoms, gdata):
        return F.cross_entropy(pred_num_atoms, gdata.num_atoms)

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss


class crysHyrbid(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = source_encoder()
        self.target_encoder = target_encoder()
        self.yprop_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.output_dim)
        self.mu_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.sigma_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.proj = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))
        self.invp = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))
        self.ce = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim = 1)
        self.pooling = AvgPooling()

    def get_indices(self, gdata, replace_ratio: float = 0.2):
        id = 0
        ids = []
        for num_atom in gdata.num_atoms:
            i = torch.randint(id, id+num_atom, (int(num_atom * replace_ratio),))
            ids.append(i)
            id+=num_atom
        return torch.cat(ids)

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
        zs = self.source_encoder(fg)
        zt = self.target_encoder(gg)
        mu, logvar = self.mu_fc(zs), self.sigma_fc(zs)
        zs = self.reparameterize(mu, logvar)


        # n_atoms = zs.shape[0]
        # indices = torch.randperm(n_atoms, device = zs.device)
        # indices = indices[:int(n_atoms*0.7)]
        # indices = self.get_indices(gprop, 0.3)
        # zh = zs.clone()
        # zh[indices,:] = zt[indices,:]

        loss_p = F.mse_loss(self.proj(zs), zt)
        loss_i = F.mse_loss(zs, self.invp(zt))
        loss_c = F.mse_loss(self.invp(self.proj(zs)), zs)
        cycle_loss = loss_p + loss_i + loss_c

        z1 = self.pooling(fg, self.proj(zs))
        z2 = self.pooling(gg[0], zt)
        # hybrid loss
        similarity = self.cos(z1, z2).mean()

        pred_prop1 = self.yprop_fc(z1)
        y1_loss = F.mse_loss(pred_prop1, yprop)

        pred_prop2 = self.yprop_fc(z2)
        y2_loss = F.mse_loss(pred_prop2, yprop)

        kld_loss = self.kld_loss(mu, logvar)
        # pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        # latt_loss = self.lattice_loss(pred_latt, gprop)
        # latt_loss*=10
        #
        # pred_comp_per_atom = self.predict_atom(zt, gprop.num_atoms)
        # atom_loss = self.atom_loss(pred_comp_per_atom, gprop)

        loss =  y2_loss + kld_loss + cycle_loss + y1_loss
        loss_dict = {
            'similarity': similarity,
            # 'hybrid_loss': hybrid_loss,
            'y1_loss': y1_loss,
            'y2_loss':y2_loss,
            'cycle_loss':cycle_loss,
            'kld_loss':kld_loss,
            # 'cl_loss': cl_loss
        }
        return loss, loss_dict

    def predice_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_length_and_angles = self.lattice_fc(z)
        scaled_preds = self.lattice_scaler.inverse_transform(pred_length_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:,3:]
        pred_lengths = pred_lengths*num_atoms.view(-1,1).float()**(1/3)
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
        # z_per_atom = z.repeat_interleave(num_atoms, dim = 0)
        pred_comp_per_atom = self.atom_fc(z)
        return pred_comp_per_atom

    def atom_loss(self, pred_comp_per_atom, gdata):
        target_atomic_num = gdata.atomic_nums -1
        loss = F.cross_entropy(pred_comp_per_atom, target_atomic_num, reduction='none')
        return scatter(loss, gdata.batch, reduce='mean').mean()
    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

class hybridTrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hybrid_encoder = hybridEncoder()
        self.pooling = AvgPooling()
        self.ce = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim=1)
    def forward(self, gg, fg, gdata, ydata):
        zs, zt, hybrid_loss = self.hybrid_encoder(gg, fg, gdata)
        z1, z2 = self.pooling(fg,zs), self.pooling(gg[0], zt)

        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gdata.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gdata)
        latt_loss *= 10

        pred_comp_per_atom = self.predict_atom(zt, gdata.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gdata)

        cos_sim = self.cos(z1,z2).mean()

        loss = hybrid_loss + latt_loss + atom_loss
        loss_dict = {
            'cos_sim': cos_sim,
            'hybrid_loss': hybrid_loss,
            'latt_loss': latt_loss,
            'atom_loss':atom_loss,
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
        # z_per_atom = z.repeat_interleave(num_atoms, dim = 0)
        pred_comp_per_atom = self.atom_fc(z)
        return pred_comp_per_atom

    def atom_loss(self, pred_comp_per_atom, gdata):
        target_atomic_num = gdata.atomic_nums - 1
        loss = F.cross_entropy(pred_comp_per_atom, target_atomic_num, reduction='none')
        return scatter(loss, gdata.batch, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
