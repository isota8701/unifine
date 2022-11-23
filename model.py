import dgl
import dgl.function as fn
import torch
from dgl.nn.pytorch import SumPooling
from utils import RBFExpansion
import torch.nn.functional as F
from torch import nn


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


class LatticeMLP(nn.Module):
    def __init__(self, args):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.hidden_features = args.hidden_features
        self.atom_embedding = MLPLayer(args.atom_input_features, args.hidden_features)
        self.module_layers = nn.ModuleList([MLPLayer(args.hidden_features, args.hidden_features) for idx in range(args.layers)])
        self.fc = nn.Linear(args.hidden_features, args.output_features)
        self.pooling = SumPooling()

    def forward(self, g):
        g = g.local_var()
        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # gated GCN updates: update node, edge features
        for module in self.module_layers:
            x = module(x)
        x = self.pooling(g,x)
        x = self.fc(x)
        return x


class AttnConv(nn.Module):

    def __init__(self, args, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.f = MLPLayer(2 * args.hidden_features, 1)
        self.g = nn.ModuleList([MLPLayer(2 * args.hidden_features, int(args.hidden_features / args.n_heads))
                                for _ in range(args.n_heads)])
        self.bn_nodes = nn.BatchNorm1d(args.hidden_features)

    def forward(self, fg: dgl.DGLGraph, x_in: torch.Tensor):
        fg = fg.local_var()

        fg.ndata['h'] = x_in
        fg.apply_edges(lambda edges: {'hihj': torch.concat([edges.src['h'], edges.dst['h']], dim=-1)})
        fg.apply_edges(lambda edges: {'exp(e_ij)': torch.exp(self.f(edges.data['hihj']))})
        fg.update_all(lambda edges: {'m':  edges.data['exp(e_ij)']}, fn.sum('m', 'sum_w_exp'))
        fg.apply_edges(lambda edges: {'a_ij': edges.data['exp(e_ij)'] / edges.dst['sum_w_exp']})
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


class FAN(nn.Module):
    def __init__(self, args):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.hidden_features = args.hidden_features
        self.atom_embedding = MLPLayer(args.atom_input_features, args.hidden_features)
        self.module_layers = nn.ModuleList([AttnConv(args) for idx in range(args.layers)])
        self.fc = nn.Linear(args.hidden_features, args.output_features)
        self.pooling = SumPooling()

    def forward(self, fg):
        fg = fg.local_var()

        # initial node features: atom feature network...
        x = fg.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # gated GCN updates: update node, edge features
        for module in self.module_layers:
            x = module(fg, x)
        x = self.pooling(fg, x)
        x = self.fc(x)
        return x