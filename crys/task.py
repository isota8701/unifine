import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from config import cfg
from gnn_2d import Encoder as source_encoder
from gnn_3d import Encoder as target_encoder
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
        self.mu_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.var_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.ce = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim = 1)

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
        mu, logvar = self.mu_fc(z2), self.var_fc(z2)
        z2 = self.reparameterize(mu, logvar)

        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gprop)
        latt_loss*=10

        pred_comp_per_atom = self.predict_atom(zn, gprop.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gprop)

        # crys_noise_cycle
        loss_p = F.mse_loss(self.proj(z1), z2)
        loss_i = F.mse_loss(z1, self.invp(z2))
        loss_c = F.mse_loss(self.invp(self.proj(z1)), z1)
        cycle_loss = loss_p + loss_i + loss_c

        kld_loss = self.kld_loss(mu, logvar)

        # crys_noise_cos
        # cos_loss = -(self.cos(self.proj(z1), z2.detach()).mean())

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

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

class crysHyrbid(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = source_encoder()
        self.target_encoder = target_encoder()
        self.lattice_fc = nn.Linear(cfg.GNN.hidden_dim, 6)
        self.lattice_scaler = torch.load(cfg.data_dir + cfg.dataset + "_LATTICE-SCALER.pt")
        self.atom_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atomic_num)
        self.mu_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
        self.var_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)
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

    def replace_feature(self, x, y, replace_prob):
        mask = torch.empty((x.size(0),), dtype = torch.float32,
                           device = x.device).uniform_(0,1) < replace_prob
        xx = x.clone()
        xx[mask,:] = y[mask,:]
        return xx


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
        # mu, logvar = self.mu_fc(zt), self.var_fc(zt)
        # zt = self.reparameterize(mu, logvar)
        z2 = self.pooling(gg[0], zt)
        mu, logvar = self.mu_fc(z2), self.var_fc(z2)
        z2 = self.reparameterize(mu, logvar)

        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gprop)

        pred_comp_per_atom = self.predict_atom(zt, gprop.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gprop)

        zh = self.replace_feature(zs, zt, 0.3)
        hybrid_loss = F.mse_loss(self.proj(zh), zt.detach())

        # loss_p = F.mse_loss(self.proj(zs), zt.detach())
        # loss_i = F.mse_loss(zs, self.invp(zt.detach()))
        # loss_c = F.mse_loss(self.invp(self.proj(zs)), zs)
        # cycle_loss = loss_p + loss_i + loss_c

        z1 = self.pooling(fg, self.proj(zs))
        # hybrid loss
        similarity = self.cos(z1, z2).mean()
        kld_loss = self.kld_loss(mu, logvar)

        loss = atom_loss + latt_loss + hybrid_loss + kld_loss
        loss_dict = {
            'similarity': similarity,
            'atom_loss': atom_loss,
            'latt_loss': latt_loss,
            'hybrid_loss': hybrid_loss,
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


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.projection = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim))

    # def forward(self, x, y):
    #     encoding_indices = self.get_code_indices(x)
    #     # print(encoding_indices[:5])
    #     quantized = self.quantize(encoding_indices)
    #     quantized = quantized.view_as(x)
    #     # embedding loss: move the embeddings towards the encoder's output
    #     q_latent_loss = F.mse_loss(quantized, x.detach())
    #     # commitment loss
    #     e_latent_loss = F.mse_loss(x, quantized.detach())
    #     s_latent_loss = F.mse_loss(y, quantized.detach())
    #     s_latent_loss+=F.mse_loss(y, x.detach())
    #     loss = q_latent_loss + self.commitment_cost * (e_latent_loss + s_latent_loss)
    #     # Straight Through Estimator
    #     quantized = (x+y) + (quantized - (x+y)).detach().contiguous()
    #
    #     return quantized, loss

    def forward(self, x,y):
        encoding_indices = self.get_code_indices(x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        e_latent_loss = F.mse_loss(x, quantized.detach())
        # y = self.projection(y)
        s_latent_loss = F.mse_loss(y, quantized.detach())
        qs_latent_loss = F.mse_loss(quantized, y.detach())
        # s_latent_loss += F.mse_loss(y, x.detach())
        loss = q_latent_loss + qs_latent_loss + self.commitment_cost*(e_latent_loss+s_latent_loss)
        quantized = (x + y) + (quantized - (x + y)).detach().contiguous()
        return quantized, loss

    def eval_forward(self, y):
        # y = self.projection(y)
        encoding_indices = self.get_code_indices(y)
        # print(encoding_indices[:5])
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(y)
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, y.detach())
        # commitment loss
        s_latent_loss = F.mse_loss(y, quantized.detach())
        loss = q_latent_loss +  self.commitment_cost*s_latent_loss
        # Straight Through Estimator
        quantized = y + (quantized - y).detach().contiguous()
        return quantized, loss
    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings.weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, self.embeddings.weight.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))

class crysVQVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.source_encoder = source_encoder()
        self.target_encoder = target_encoder()
        self.vq_layer = VectorQuantizer(embedding_dim=cfg.GNN.hidden_dim,
                                        num_embeddings=512,
                                        commitment_cost=0.25)

        self.lattice_fc = nn.Linear(cfg.GNN.hidden_dim, 6)
        self.lattice_scaler = torch.load(cfg.data_dir + cfg.dataset + "_LATTICE-SCALER.pt")
        self.atom_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atomic_num)
        self.num_fc = nn.Linear(cfg.GNN.hidden_dim, cfg.max_atoms+1)
        self.pooling = AvgPooling()

    def forward(self, gg, fg, gprop, yprop):
        zs = self.source_encoder(fg)
        zt = self.target_encoder(gg)

        zq, vq_loss = self.vq_layer(zt, zs)
        z2 = self.pooling(gg[0], zq)


        pred_latt, pred_lengths, pred_angles = self.predice_lattice(z2, gprop.num_atoms)
        latt_loss = self.lattice_loss(pred_latt, gprop)
        # latt_loss*=10

        pred_comp_per_atom = self.predict_atom(zq, gprop.num_atoms)
        atom_loss = self.atom_loss(pred_comp_per_atom, gprop)


        loss =  vq_loss + atom_loss + latt_loss
        loss_dict = {
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

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss
