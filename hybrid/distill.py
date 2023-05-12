import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from config import cfg
from gnn_2d import Encoder as source_encoder
from gnn_3d import Encoder as target_encoder
import numpy as np


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

        # hybrid loss
        # replace_ratio = 1-(self.cos(z1,z2).mean())
        # indices = self.get_indices(gprop, replace_ratio)
        # zhybrid = zs.clone()
        # zhybrid[indices,:] = zt[indices,:]
        # zhybrid = self.projection(zhybrid)
        # hybrid_loss = F.mse_loss(zhybrid, zt)

        # y loss
        pred_prop = self.yprop_fc(z2)
        y_loss = F.mse_loss(pred_prop, yprop)


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
