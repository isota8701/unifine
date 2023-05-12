import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from config import cfg
from torch import Tensor
from torch_sparse import SparseTensor, set_diag


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
    def __init__(self, residual: bool = True):
        super().__init__(node_dim=0)
        self.residual = residual
        self.heads = cfg.FORMULA.n_heads
        self.hidden_dim = cfg.FORMULA.hidden_dim
        self.lin_src = nn.Linear(self.hidden_dim, self.heads * self.hidden_dim, bias=False)
        self.lin_dst = self.lin_src
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.edge_dim = None
        self.add_self_loops = True
        self.fill_value = 'mean'

        self.att_src = nn.Parameter(torch.Tensor(1, self.heads, self.hidden_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, self.heads, self.hidden_dim))
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
        self.bias = nn.Parameter(torch.Tensor(self.heads * self.hidden_dim))
        torch.nn.init.zeros_(self.bias)

        self.bn_nodes = nn.BatchNorm1d(self.heads * self.hidden_dim)
        self.lin_out = nn.Linear(self.heads * self.hidden_dim, self.hidden_dim, bias=False)

    def forward(self, x_in, edge_index, atom_w):
        H, C = self.heads, self.hidden_dim
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
        out = out.view(-1, self.heads * self.hidden_dim)
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
    def __init__(self):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.hidden_dim = cfg.FORMULA.hidden_dim
        self.atom_embedding = MLPLayer(cfg.FORMULA.atom_input_dim, self.hidden_dim)
        self.module_layers = nn.ModuleList([RConv() for idx in range(cfg.FORMULA.layers)])

    def forward(self, data):
        x, e, w, b = data.node_features_s, data.edge_index_s, data.atom_weights_s, data.atom_types_s_batch
        x = self.atom_embedding(x)
        for module in self.module_layers:
            x = module(x, e, w)
        x = global_add_pool(x, b)
        return x
