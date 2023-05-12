import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, GINConv
from config import cfg
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Parameter
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, PairTensor
from typing import Optional, Union
from torch import Tensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_sparse import SparseTensor, set_diag

class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   nn.SiLU())
    def forward(self, x):
        return self.layer(x)





class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    def __init__(
            self,
            in_channels= cfg.FORMULA.hidden_dim,
            out_channels= cfg.FORMULA.hidden_dim,
            heads= cfg.FORMULA.n_heads,
            concat: bool = False,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = 256,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        self.bn_nodes = nn.BatchNorm1d(self.out_channels)
        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
        torch.nn.init.xavier_uniform_(self.att_edge)
        torch.nn.init.zeros_(self.bias)

    def forward(self,x_in: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x_in, Tensor):
            assert x_in.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x_in).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x_in
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
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

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        out = self.bn_nodes(out)
        out += x_in
        return out


    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


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


class CGConv(MessagePassing):
    r"""The crystal graph convolutional operator from the
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_
    paper


    Args:
        channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        dim (int, optional): Edge feature dimensionality. (default: :obj:`0`)
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        batch_norm (bool, optional): If set to :obj:`True`, will make use of
            batch normalization. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)` or
          :math:`(|\mathcal{V_t}|, F_{t})` if bipartite
    """
    def __init__(self ):
        super().__init__(aggr='add')
        self.channels = cfg.FORMULA.hidden_dim
        bias = True

        self.lin_f = Linear(self.channels*2, self.channels, bias=bias)
        self.lin_s = Linear(self.channels*2, self.channels, bias=bias)
        self.lin_e = Linear(self.channels, self.channels, bias = bias)
        self.bn = nn.BatchNorm1d(self.channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        self.lin_e.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # edge_attr = self.lin_e(edge_attr)
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out if self.bn is None else self.bn(out)
        out = out + x[1]
        return out


    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels})'

class FormulaNet(nn.Module):
    def __init__(self):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.hidden_dim = cfg.FORMULA.hidden_dim
        self.atom_embedding = MLPLayer(cfg.FORMULA.atom_input_dim, self.hidden_dim)
        self.module_layers = nn.ModuleList([GATConv() for idx in range(cfg.FORMULA.layers)])
        self.device = cfg.DEVICE
        self.fc_mu = nn.Linear(cfg.FORMULA.hidden_dim,
                               cfg.FORMULA.hidden_dim)
        self.fc_var = nn.Linear(cfg.FORMULA.hidden_dim,
                                cfg.FORMULA.hidden_dim)
        # to test formula net
        self.out_fc = nn.Linear(self.hidden_dim, 1)

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

    def forward(self, data):
        x, e, w = data.node_features_s, data.edge_index_s, data.atom_weights_s
        # if "atom_types_s_batch" in data.keys:
        #     batch = data.atom_types_s_batch
        # elif "batch" in data.keys:
        #     batch = data.batch
        x = self.atom_embedding(x)
        for module in self.module_layers:
            x = module(x,e)
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        # z = self.reparameterize(mu, logvar)
        # to test formula net
        # x = global_add_pool(x, batch)
        # x = self.out_fc(x)
        return x, mu, logvar

'''
class DenoisingFormula

x
x = atom_emb(x)
x = GIN_1(x)
z = repara(mu,var)
zz = z+noise
n = GIN_2(zz)

pretrain.py
loss(n, gt)
gt = 3d_net(3dx) --> repara : z_3d - n (gaussian diff) or simple mse


'''

if __name__ == "__main__":
    from pretrain_cl.data import MaterialLoader
    import numpy as np
    train_loader, valid_loader, test_loader = MaterialLoader()
    model = FormulaNet()
    model = model.to(cfg.DEVICE)
    from torch import optim

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(),
                           lr=cfg.TRAIN.lr)

    for e in range(cfg.TRAIN.max_epoch):
        running_loss = []
        for batch in train_loader:
            batch.to(cfg.DEVICE)
            outs = model(batch)
            loss = loss_fn(outs, batch.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss.append(loss.item())
        if (e+1) % cfg.TRAIN.snapshot_interval == 0:
            print(f"epoch {e+1}, loss: {np.mean(running_loss):.3f}")
    if (e+1) == cfg.TRAIN.max_epoch:
        print(f"finished")
        mae_fn = nn.L1Loss()
        model.eval()
        running_loss = []
        with torch.no_grad():
            for batch in test_loader:
                batch.to(model.device)
                outputs = model(batch)
                mae = mae_fn(outputs, batch.y)
                running_loss.append(mae.item())
            print(f"{np.mean(running_loss): .3f}")