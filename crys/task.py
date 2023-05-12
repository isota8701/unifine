import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from config import cfg
from gnn_2d import Encoder as source_encoder
from gnn_3d import Encoder as target_encoder
import numpy as np

class crysCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = source_encoder()
        self.target_encoder = target_encoder()
        self.yprop_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.output_dim)
        self.lattice_fc = nn.Linear(cfg.GNN.hidden_dim, 6)
        self.lattice_scaler = torch.load(cfg.data_dir + cfg.dataset + "_LATTICE-SCALER.pt")
        self.atom_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atomic_num)
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
        z1 = self.source_encoder(fg)
        zn, z2 = self.target_encoder(gg)

        pred_prop = self.yprop_fc(z2)
        # y_loss = F.mse_loss(pred_prop, yprop)

        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gprop)
        latt_loss*=10

        pred_comp_per_atom = self.predict_atom(zn, gprop.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gprop)

        #info nce loss
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
        # cl_loss = self.ce(logits, labels)

        # clip loss
        # z1 = z1 @ self.source_proj
        # z2 = z2 @ self.target_proj
        # z1 = z1/z1.norm(dim = 1, keepdim= True)
        # z2 = z2/z2.norm(dim=1, keepdim= True)
        #
        # logit_scale = self.logit_scale.exp()
        # logits_per = logit_scale* z1 @ z2.T
        # lbl = torch.arange(z1.shape[0]).to(z1.device)
        # clip_loss = (self.ce(logits_per, lbl) + self.ce(logits_per.T,lbl))/2

        loss_p = F.mse_loss(self.proj(z1), z2)
        loss_i = F.mse_loss(z1, self.invp(z2))
        loss_c = F.mse_loss(self.invp(self.proj(z1)), z1)
        cycle_loss = loss_p + loss_i + loss_c


        # cos_loss = -(self.cos(self.proj(z1), z2).mean())
        loss = cycle_loss + atom_loss + latt_loss
        loss_dict = {
            # 'y_loss': y_loss,
            # 'cos_loss': cos_loss,
            'atom_loss': atom_loss,
            # 'cl_loss': cl_loss,
            'cycyle_loss': cycle_loss,
            'latt_loss':latt_loss,
            # 'kl_loss':kl_loss
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

#EM style


class target_solver(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_encoder = target_encoder()
        self.yprop_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.output_dim)
        self.lattice_fc = nn.Linear(cfg.GNN.hidden_dim, 6)
        self.lattice_scaler = torch.load(cfg.data_dir + cfg.dataset + "_LATTICE-SCALER.pt")
        self.atom_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atomic_num)
        self.atom_token = nn.Parameter(torch.randn(1,cfg.GNN.hidden_dim))
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, gg, gprop, yprop):
        zn, z2 = self.target_encoder(gg)

        pred_prop = self.yprop_fc(z2)
        y_loss = self.mse(pred_prop, yprop)

        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gprop)

        pred_comp_per_atom = self.predict_atom(zn, gprop.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gprop)

        latt_loss*=10

        loss = latt_loss + atom_loss
        loss_dict = {
            # 'y_loss': y_loss,
            'atom_loss': atom_loss,
            # 'cl_loss': cl_loss,
            'latt_loss':latt_loss,
        }
        return z2, loss, loss_dict

    def predice_lattice(self, z, num_atoms):
        # w/ lattice scaling
        self.lattice_scaler.match_device(z)
        pred_length_and_angles = self.lattice_fc(z)
        scaled_preds = self.lattice_scaler.inverse_transform(pred_length_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:,3:]
        pred_lengths = pred_lengths*num_atoms.view(-1,1).float()**(1/3)

        #w/o lattice scaling
        # self.lattice_scaler.match_device(z)
        # pred_length_and_angles = self.lattice_fc(z)
        # pred_lengths = pred_length_and_angles[:, :3]
        # pred_angles = pred_length_and_angles[:, 3:]
        # pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)

        return pred_length_and_angles, pred_lengths, pred_angles

    def lattice_loss(self, pred_lengths_and_angles, gdata):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        # w lattice scaling
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

    def info_nce_loss(self, z1, z2):

        z = torch.cat([z1,z2], dim = 0)
        z = F.normalize(z, dim = -1)
        sim_mat = torch.matmul(z, z.T)

        lbl = torch.cat([torch.arange(z1.shape[0]) for _ in range(2)], dim = 0)
        lbl = (lbl.unsqueeze(0) == lbl.unsqueeze(1)).float()
        lbl = lbl.to(z.device)

        mask = torch.eye(lbl.shape[0], dtype = torch.bool).to(z.device)

        lbl = lbl[~mask].view(lbl.shape[0],-1)
        sim_mat = sim_mat[~mask].view(sim_mat.shape[0],-1)

        positives = sim_mat[lbl.bool()].view(lbl.shape[0],-1)
        negatives = sim_mat[~lbl.bool()].view(sim_mat.shape[0],-1)
        logits = torch.cat([positives, negatives], dim = 1)
        logits = logits / cfg.temperature
        labels = torch.zeros(logits.shape[0], dtype = torch.long).to(z.device)
        cl_loss = self.ce(logits, labels)
        return cl_loss

class supervised_baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = source_encoder()
        self.target_encoder = target_encoder()
        self.proj = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))
        self.y_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.output_dim)

    def forward(self, gg, fg, gprop, yprop):
        z1 = self.source_encoder(fg)
        z2 = self.target_encoder(gg)
        if cfg.weights == 'supervised':
            ypred = self.y_fc(z1)
        else:
            raise "weights not supervised"
        yloss = F.mse_loss(yprop, ypred)
        loss_dict = {'y2_loss': yloss}
        return yloss, loss_dict



class crysCycle(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = source_encoder()
        self.target_encoder = target_encoder()
        self.yprop_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.output_dim)
        self.lattice_fc = nn.Linear(cfg.GNN.hidden_dim, 6)
        self.lattice_scaler = torch.load(cfg.data_dir + cfg.dataset + "_LATTICE-SCALER.pt")
        self.atom_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atomic_num)
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
        z1 = self.source_encoder(fg)
        _, z2 = self.target_encoder(gg)
        loss_p = F.mse_loss(self.proj(z1), z2)
        loss_i = F.mse_loss(z1, self.invp(z2))
        loss_c = F.mse_loss(self.invp(self.proj(z1)), z1)
        cycle_loss = loss_p + loss_i + loss_c
        ypred1 = self.yprop_fc(self.proj(z1))
        y1_loss = F.mse_loss(yprop, ypred1)
        ypred2 = self.yprop_fc(z2)
        y2_loss = F.mse_loss(yprop, ypred2)


        cos_loss = -(self.cos(self.proj(z1), z2).mean())

        # pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        # latt_loss = self.lattice_loss(pred_latt, gprop)
        # latt_loss*=10
        #
        # pred_comp_per_atom = self.predict_atom(zn, gprop.num_atoms)
        # atom_loss = self.atom_loss(pred_comp_per_atom, gprop)

        loss = y1_loss + y2_loss + cos_loss
        loss_dict = {
            # 'cycle_loss': cycle_loss,
            # 'atom_loss': atom_loss,
            # 'latt_loss': latt_loss,
            'cosine_loss': cos_loss,
            'y2_loss': y2_loss,
            'y1_loss': y1_loss
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

class crysVAE(nn.Module):
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
        self.proj = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))
        self.invp = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))
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
        loss_p = F.mse_loss(self.proj(z1), z2)
        loss_i = F.mse_loss(z1, self.invp(z2))
        loss_c = F.mse_loss(self.invp(self.proj(z1)), z1)
        cycle_loss = loss_p + loss_i + loss_c

        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gprop)
        latt_loss *= 10

        pred_comp_per_atom = self.predict_atom(zn, gprop.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gprop)

        kld_loss = self.kld_loss(mu, logvar)

        loss = cycle_loss + atom_loss + latt_loss + kld_loss
        loss_dict = {
            'cycle_loss': cycle_loss,
            'atom_loss': atom_loss,
            'latt_loss': latt_loss,
            'kld_loss': kld_loss,
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
        return kld_loss

class crysVQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = source_encoder()
        self.target_encoder = target_encoder()
        self.embeddings = nn.Embedding(cfg.max_atomic_num, 256)
        self.yprop_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.output_dim)
        self.lattice_fc = nn.Linear(cfg.GNN.hidden_dim, 6)
        self.lattice_scaler = torch.load(cfg.data_dir + cfg.dataset + "_LATTICE-SCALER.pt")
        self.atom_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atomic_num)
        self.proj = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))
        self.invp = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))

    def get_code_indicies(self, flat_x):
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) +
                     torch.sum(self.embeddings.weight**2, dim=1) -
                     2.*torch.matmul(flat_x, self.embeddings.weight.t())) # [N,M]
        encoding_indices = torch.argmin(distances, dim = 1) #[N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        return self.embeddings(encoding_indices)

    def vq_forward(self, x):
        encoding_indices = self.get_code_indicies(x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + e_latent_loss
        quantized = x + (quantized - x).detach().contiguous()
        return quantized, loss

    def forward(self, gg, fg, gprop, yprop):
        z1 = self.source_encoder(fg)
        zn, z2 = self.target_encoder(gg)
        z2, vq_loss = self.vq_forward(z2)

        loss_p = F.mse_loss(self.proj(z1), z2)
        loss_i = F.mse_loss(z1, self.invp(z2))
        loss_c = F.mse_loss(self.invp(self.proj(z1)), z1)
        cycle_loss = loss_p + loss_i + loss_c

        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gprop)
        latt_loss *= 10

        pred_comp_per_atom = self.predict_atom(zn, gprop.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gprop)


        loss = cycle_loss + atom_loss + latt_loss + vq_loss
        loss_dict = {
            'cycle_loss': cycle_loss,
            'atom_loss': atom_loss,
            'latt_loss': latt_loss,
            'vq_loss': vq_loss,
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
        return kld_loss



class crysHyrbid(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = source_encoder()
        self.target_encoder = target_encoder()
        self.yprop_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.output_dim)
        self.projection = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))

        self.ce = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim = 1)

    def get_indices(self, gdata, replace_ratio: float = 0.2):
        id = 0
        ids = []
        for num_atom in gdata.num_atoms:
            i = torch.randint(id, id+num_atom, (int(num_atom * replace_ratio),))
            ids.append(i)
            id+=num_atom
        return torch.cat(ids)



    def forward(self, gg, fg, gprop, yprop):
        zs, z1 = self.source_encoder(fg)
        zt, z2 = self.target_encoder(gg)



        # base line 1 target
        pred_prop = self.yprop_fc(z2)
        y_loss = F.mse_loss(pred_prop, yprop)

        # base line 2 source
        pred_prop = self.yprop_fc(z1)
        y_loss = F.mse_loss(pred_prop, yprop)

        # hybrid loss
        # replace_ratio = 1-(self.cos(z1,z2).mean())
        # indices = self.get_indices(gprop, replace_ratio)
        # zhybrid = zs.clone()
        # zhybrid[indices,:] = zt[indices,:]
        # zhybrid = self.projection(zhybrid)
        # hybrid_loss = F.mse_loss(zhybrid, zt)



        # pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        # latt_loss = self.lattice_loss(pred_latt, gprop)
        # latt_loss*=10
        #
        # pred_comp_per_atom = self.predict_atom(zt, gprop.num_atoms)
        # atom_loss = self.atom_loss(pred_comp_per_atom, gprop)

        loss =  y_loss
        loss_dict = {
            # 'hybrid_loss': hybrid_loss,
            'y_loss': y_loss,
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
