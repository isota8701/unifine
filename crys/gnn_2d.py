import dgl
import dgl.function as fn
import torch
from dgl.nn.pytorch.glob import AvgPooling, SumPooling
from torch import nn
from config import cfg


class MLPLayer(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(nn.Linear(in_dim, out_dim),
                         nn.BatchNorm1d(out_dim),
                         nn.SiLU()
                         )

class Roost(nn.Module):
    def __init__(self,
                 hidden_dim: int = cfg.GNN.hidden_dim,
                 n_heads : int = cfg.GNN.num_heads,
                 residual: bool = True):
        super().__init__()
        self.residual = residual
        self.f = MLPLayer(2 * hidden_dim, 1)  # unnormalised scalar coefficient
        self.g = nn.ModuleList([MLPLayer(2 * hidden_dim, int(hidden_dim / n_heads))
                                for _ in range(n_heads)])
        self.bn_nodes = nn.BatchNorm1d(hidden_dim)

    def forward(self, fg: dgl.DGLGraph, x_in: torch.Tensor):
        fg = fg.local_var()

        fg.ndata['h'] = x_in
        fg.apply_edges(lambda edges: {'hihj': torch.concat([edges.src['h'], edges.dst['h']], dim=-1)})
        fg.apply_edges(lambda edges: {'exp(e_ij)': torch.exp(self.f(edges.data['hihj']))})
        fg.update_all(lambda edges: {'m': edges.src['weight'] * edges.data['exp(e_ij)']}, fn.sum('m', 'sum_w_exp'))
        fg.apply_edges(lambda edges: {'a_ij': edges.src['weight'] * edges.data['exp(e_ij)'] / edges.dst['sum_w_exp']})
        fg.update_all(self.message, fn.sum('m', 'h'))
        x = self.bn_nodes(fg.ndata['h'])
        if self.residual:
            x = x + x_in
        return x

    def message(self, edges):
        head_feat = []
        for g in self.g:
            head_feat.append(edges.data['a_ij'] * g(edges.data['hihj']))

        return {'m': torch.concat(head_feat, dim=-1)}


class Encoder(nn.Module):
    def __init__(self,
                 atom_input_dim: int = cfg.GNN.atom_input_dim,
                 hidden_dim: int= cfg.GNN.hidden_dim,
                 roost_layers: int = cfg.GNN.roost_layers,
                 ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.hidden_features = hidden_dim
        self.atom_embedding = MLPLayer(atom_input_dim, hidden_dim)
        self.module_layers = nn.ModuleList([Roost() for _ in range(roost_layers)])
        self.pooling = AvgPooling()

    def forward(self, fg):
        fg = fg.local_var()
        # initial node features: atom feature network...
        x = fg.ndata.pop("atom_features")
        x = self.atom_embedding(x)
        # gated GCN updates: update node, edge features
        for module in self.module_layers:
            x = module(fg, x)
        xr = self.pooling(fg, x)
        return xr
