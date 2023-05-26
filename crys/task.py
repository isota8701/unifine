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
        self.source_encoder = source_encoder() # noise inj x -> enc -> readout(x)
        self.target_encoder = target_encoder() # noise inj x -> enc -> x, readout(x)
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
        z1 = self.source_encoder(fg)
        zn, z2 = self.target_encoder(gg)

        # crys_noise_cycle
        loss_p = F.mse_loss(self.proj(z1), z2)
        loss_i = F.mse_loss(z1, self.invp(z2))
        loss_c = F.mse_loss(self.invp(self.proj(z1)), z1)
        cycle_loss = loss_p + loss_i + loss_c

        # crys_noise_cos
        # cos_loss = -(self.cos(self.proj(z1), z2).mean())

        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gprop)
        latt_loss*=10

        pred_comp_per_atom = self.predict_atom(zn, gprop.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gprop)

        loss =  cycle_loss + atom_loss + latt_loss
        loss_dict = {
            'cycle_loss': cycle_loss,
            'atom_loss': atom_loss,
            'latt_loss': latt_loss,
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


