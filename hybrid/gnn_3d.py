import dgl
import dgl.function as fn
import torch
from dgl.nn.pytorch.glob import AvgPooling,SumPooling
from utils import RBFExpansion
import torch.nn.functional as F
from torch import nn
from config import cfg


class EGGConv(nn.Module):

    def __init__(
        self, input_features: int, output_features: int):
        super().__init__()
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.
        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        return x, m


class Conv(nn.Module):
    def __init__(self, in_feat, out_feat, residual=True):
        super().__init__()
        self.residual = residual
        self.gnn = EGGConv(in_feat, out_feat)
        self.bn_nodes = nn.BatchNorm1d(out_feat)
        self.bn_edges = nn.BatchNorm1d(out_feat)

    def forward(self, g, x_in, y_in):
        x, y = self.gnn(g, x_in, y_in)
        x, y = F.silu(self.bn_nodes(x)), F.silu(self.bn_edges(y))

        if self.residual:
            x = x + x_in
            y = y + y_in

        return x, y


class ALIGNNConv(nn.Module):
    def __init__(self,
                 line_graph : bool,
                 hidden_dim: int = cfg.GNN.hidden_dim,
                 residual: bool = True):
        super().__init__()
        self.line_graph = line_graph
        self.node_update = Conv(hidden_dim, hidden_dim, residual)
        if self.line_graph:
            self.edge_update = Conv(hidden_dim, hidden_dim, residual)

    def forward(self, g, lg, x_in, y_in, z_in):
        g = g.local_var()

        x, y = self.node_update(g, x_in, y_in)
        z = z_in
        if self.line_graph:
            lg = lg.local_var()
            # Edge-gated graph convolution update on crystal graph
            y, z = self.edge_update(lg, y, z)

        return x, y, z


class MLPLayer(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(nn.Linear(in_dim, out_dim),
                         nn.BatchNorm1d(out_dim),
                         nn.SiLU()
                         )

class Encoder(nn.Module):
    def __init__(self,
                 atom_input_dim: int = cfg.GNN.atom_input_dim,
                 edge_input_dim: int = cfg.GNN.edge_input_dim,
                 triplet_input_dim: int = cfg.GNN.triplet_input_dim,
                 embedding_dim: int = cfg.GNN.embedding_dim,
                 hidden_dim: int = cfg.GNN.hidden_dim,
                 alignn_layers: int = cfg.GNN.alignn_layers,
                 gcn_layers: int = cfg.GNN.gcn_layers):

        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.hidden_features = hidden_dim
        self.atom_embedding = MLPLayer(atom_input_dim, hidden_dim)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8, bins=edge_input_dim),
            MLPLayer(edge_input_dim, embedding_dim),
            MLPLayer(embedding_dim, hidden_dim),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_dim),
            MLPLayer(triplet_input_dim, embedding_dim),
            MLPLayer(embedding_dim, hidden_dim),
        )
        self.module_layers = nn.ModuleList([ALIGNNConv(line_graph=True) for _ in range(alignn_layers)]
                                           + [ALIGNNConv(line_graph=False) for _ in range(gcn_layers)])
        self.pooling = AvgPooling()

    def forward(self, gg):
        g = gg[0]
        lg = gg[1]
        g = g.local_var()
        lg = lg.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        y = self.edge_embedding(g.edata['d'])

        # angle features (fixed)
        z = self.angle_embedding(lg.edata.pop("h"))
        # gated GCN updates: update node, edge features
        for module in self.module_layers:
            x, y, z = module(g, lg, x, y, z)
        xr = self.pooling(g,x)
        return x, xr



