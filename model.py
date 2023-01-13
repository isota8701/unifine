# import dgl
# import dgl.function as fn
# from dgl.nn.pytorch import SumPooling
# from utils import RBFExpansion
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add, scatter_max
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool
from dataset import LoadDataset
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from utils import frac_to_cart_coords, min_distance_sqr_pbc, radius_graph_pbc, get_pbc_distances, repeat_blocks, MAX_ATOMIC_NUM
from torch_scatter import scatter
from gemnet import GemNetT as Decoder
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
# class AttnConv(nn.Module):
#
#     def __init__(self, args, residual: bool = True):
#         super().__init__()
#         self.residual = residual
#         self.f = MLPLayer(2 * args.hidden_features, 1)
#         self.g = nn.ModuleList([MLPLayer(2 * args.hidden_features, int(args.hidden_features / args.n_heads))
#                                 for _ in range(args.n_heads)])
#         self.bn_nodes = nn.BatchNorm1d(args.hidden_features)
#
#     def forward(self, fg: dgl.DGLGraph, x_in: torch.Tensor):
#         fg = fg.local_var()
#
#         fg.ndata['h'] = x_in
#         fg.apply_edges(lambda edges: {'hihj': torch.concat([edges.src['h'], edges.dst['h']], dim=-1)})
#         fg.apply_edges(lambda edges: {'exp(e_ij)': torch.exp(self.f(edges.data['hihj']))})
#         fg.update_all(lambda edges: {'m':  edges.data['exp(e_ij)']}, fn.sum('m', 'sum_w_exp'))
#         fg.apply_edges(lambda edges: {'a_ij': edges.data['exp(e_ij)'] / edges.dst['sum_w_exp']})
#         fg.update_all(self.message, fn.sum('m', 'h'))

#         if self.residual:
#             x = x + x_in
#         return x
#     def message(self, edges):
#         head_feat = []
#         for g in self.g:
#             head_feat.append(edges.data['a_ij'] * g(edges.data['hihj']))
#
#         return {'m': torch.concat(head_feat, dim=-1)}

# class LatticeMLP(nn.Module):
#     def __init__(self, args):
#         """Initialize class with number of input features, conv layers."""
#         super().__init__()
#
#         self.hidden_features = args.hidden_features
#         self.atom_embedding = MLPLayer(args.atom_input_features, args.hidden_features)
#         self.module_layers = nn.ModuleList([MLPLayer(args.hidden_features, args.hidden_features) for idx in range(args.layers)])
#         self.fc = nn.Linear(args.hidden_features, args.output_features)
#         self.pooling = SumPooling()
#
#     def forward(self, g):
#         g = g.local_var()
#         # initial node features: atom feature network...
#         x = g.ndata.pop("atom_features")
#         x = self.atom_embedding(x)
#
#         # gated GCN updates: update node, edge features
#         for module in self.module_layers:
#             x = module(x)
#         x = self.pooling(g,x)
#         x = self.fc(x)
#         return x
class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class RConv(MessagePassing):
    def __init__(self, args, residual: bool = True):
        super().__init__(node_dim=0)
        self.residual = residual
        self.heads = args.n_heads
        self.lin_src = nn.Linear(args.hidden_features, self.heads * args.hidden_features, bias=False)
        self.lin_dst = self.lin_src
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.edge_dim = None
        self.add_self_loops = True
        self.fill_value = 'mean'

        self.att_src = nn.Parameter(torch.Tensor(1, self.heads, args.hidden_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, self.heads, args.hidden_features))
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
        self.bias = nn.Parameter(torch.Tensor(self.heads * args.hidden_features))
        torch.nn.init.zeros_(self.bias)

        self.bn_nodes = nn.BatchNorm1d(self.heads * args.hidden_features)
        self.lin_out = nn.Linear(self.heads * args.hidden_features, args.hidden_features, bias=False)

    def forward(self, x_in, edge_index, atom_w):
        H, C = self.heads, args.hidden_features
        x_src = x_dst = self.lin_src(x_in).view(-1, H, C)
        x = (x_src, x_dst)
        weight = (atom_w, atom_w)
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr = None)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")
        alpha = self.edge_updater(edge_index, alpha=alpha, weight=weight)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out.view(-1, self.heads * args.hidden_features)
        out += self.bias
        out = self.bn_nodes(out)
        out = self.lin_out(out)
        if self.residual:
            out = out + x_in
        return out

    def edge_update(self, alpha_j, alpha_i, weight_j, index, ptr, size_i):
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha)
        alpha *= weight_j
        alpha = softmax(alpha, index, ptr, size_i)
        return alpha

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j


class FormulaNet(nn.Module):
    def __init__(self, args):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.hidden_features = args.hidden_features
        self.atom_embedding = MLPLayer(args.atom_input_features, args.hidden_features)
        self.module_layers = nn.ModuleList([RConv(args) for idx in range(args.layers)])

    def forward(self, data):
        x, e, w, b = data.node_features, data.edge_index, data.atom_weights, data.batch
        x = self.atom_embedding(x)
        for module in self.module_layers:
            x = module(x, e, w)
        readout_x = global_add_pool(x, b)
        return x, readout_x


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class latticeVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        '''
        1. self.encoder input : formula graph output: latent z
        2. reparametrize
        3. predict lattice, num atoms, composition
        4. self.decoder input: latent, predicted lattice output: edges
        '''
        self.encoder = FormulaNet(args)
        self.fc_mu = nn.Linear(args.hidden_features,
                               args.hidden_features)
        self.fc_var = nn.Linear(args.hidden_features,
                                args.hidden_features)

        self.fc_num_atoms = build_mlp(args.hidden_features, args.hidden_features, 3, args.max_atoms)
        self.fc_lattice = build_mlp(args.hidden_features, args.hidden_features, 3, 6)
        self.fc_composition = build_mlp(args.hidden_features, args.hidden_features, 3, MAX_ATOMIC_NUM)
        self.lattice_scaler = torch.load("./data/" + "lattice_scaler.pt")

    def encode(self, batch):
        n_hidden, hidden= self.encoder(batch)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        # parameter share or not?
        n_mu = self.fc_mu(n_hidden)
        n_log_var = self.fc_var(n_hidden)
        n_z = self.reparameterize(n_mu, n_log_var)
        return mu, log_var, z, n_z

    def decode_stats(self, z, gt_num_atoms):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing
        """
        num_atoms = self.predict_num_atoms(z)
        lengths_and_angles, lengths, angles = (
            self.predict_lattice(z, gt_num_atoms))
        composition_per_atom = self.predict_composition(z, gt_num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

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

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))

    def predict_lattice(self, z, num_atoms):
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        self.lattice_scaler.match_device(z)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom


    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

    def forward(self, batch):
        mu, log_var, z, n_z= self.encode(batch)
        decode_stat =  self.decode_stats(z, gt_num_atoms=batch.num_atoms)
        # kld loss for each n mu log_var?
        kld_loss = 0.001*self.kld_loss(mu, log_var)
        return z,n_z, decode_stat, kld_loss


class latticeDEC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.decoder = Decoder()
        self.fc_coord = build_mlp(args.hidden_features, args.hidden_features, 3, 3)
        self.fc_atom = nn.Linear(512,MAX_ATOMIC_NUM )
        self.device = args.device
    def predict_coord(self, n_z, num_atoms):
        z_per_atom = n_z.repeat_interleave(num_atoms, dim=0)
        pred_coords_per_atom = self.fc_coord(z_per_atom)
        return pred_coords_per_atom
    def forward(self, z, decode_stats, batch):
        rep_type = torch.LongTensor().to(self.device)
        for i in range(len(batch)):
            b = batch[i]
            cnt = torch.unique_consecutive(b.target_atom_types, return_counts = True)[1]
            rep_type = torch.cat((rep_type, cnt))
        assert rep_type.sum() == batch.target_num_atoms.sum()
        pseudo_cart_coord = self.predict_coord(z,rep_type)

        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles, pred_composition_per_atom) = decode_stats
        # what atom type, 3D? or predicted atom type or 2D <- use 2D,

        h,  pred_cart_coord= self.decoder(
        z, pseudo_cart_coord, batch.atom_types, batch.num_atoms, pred_lengths, pred_angles)
        pred_atom_types = self.fc_atom(h)
        return pred_cart_coord, pred_atom_types

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=' prediction training')
    parser.add_argument('--layers', type=int, default=4, help="")
    parser.add_argument('--atom-input-features', type=int, default=92, help="")
    parser.add_argument('--hidden-features', type=int, default=256, help="")
    parser.add_argument('--output-features', type=int, default=9, help="")
    parser.add_argument('--n-heads', type=int, default=6, help="")
    parser.add_argument('--dataset', type=str, default='mp_3d_2020')
    parser.add_argument('--max-atoms', type = int, default= 20)
    parser.add_argument('--num-train', type=int, default=100, help="")
    parser.add_argument('--num-valid', type=int, default=25, help="")
    parser.add_argument('--num-test', type=int, default=25, help="")
    parser.add_argument('--batch-size', type=int, default=20, help="")
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--alpha', type = float, default = 1.)
    parser.add_argument('--beta', type = float, default=10.)
    parser.add_argument('--gamma', type = float, default=1.)
    parser.add_argument('--device', type=str, default='cuda:0', help="cuda device")
    args = parser.parse_args()
    from dataset import MaterialLoader
    train_loader, valid_loader, test_loader = MaterialLoader(args)
    model = latticeVAE(args)
    model = model.to(args.device)
    lattDec = latticeDEC(args)
    lattDec = lattDec.to(args.device)
    from loss import LatticeLoss
    from torch import optim
    import numpy as np
    optimizer = optim.Adam(params= model.parameters(),
                           lr = 0.0001)

    for e in range(100):
        cnt = 0
        loss = []
        for batch in train_loader:
            batch.to(args.device)
            z,n_z, decode_stat, kld_loss= model(batch)
            lattLoss = LatticeLoss(args)
            loss1 = lattLoss(decode_stat, batch)
            loss1+=kld_loss
            loss1.backward()
            optimizer.step()
            optimizer.zero_grad()
            cnt+=1
            loss.append(loss1.item())
        print(f"epoch {e+1}, loss: {np.mean(loss):.3f}")
        if e+1 == 100:
            h,F = lattDec(n_z, decode_stat,batch)
