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

