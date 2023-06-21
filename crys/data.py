from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.core.specie import get_node_attributes
from jarvis.core.graphs import compute_bond_cosines
from jarvis.db.figshare import data as jdata
from ase import neighborlist
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import dgl
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import re
from config import cfg
from utils import get_scaler_from_data_tensor


def compute_d_u(edges):
    r = edges.dst['pos'] - edges.src['pos']
    d = torch.norm(r, dim=1)
    u = r / d[:, None]
    return {'r': r, 'd': d, 'u': u}


def json_dict_to_graph(dict, cutoff, max_neighbors, pbc=False):
    structure = Atoms.from_dict(dict)
    mol = structure.ase_converter(pbc=pbc)
    g, lg = Graph.atom_dgl_multigraph(structure, cutoff=cutoff, use_canonize=False, max_neighbors=max_neighbors)
    g.ndata['pos'] = torch.tensor(mol.get_positions(), dtype=torch.float32)
    g.apply_edges(lambda edges: {'d': torch.norm(g.edata['r'], dim=1)})
    # 3d encoder graph level task
    num_atoms = mol.get_global_number_of_atoms()
    lattice = torch.tensor(mol.cell.cellpar(), dtype = torch.float32).view(1,-1)
    lengths = lattice[:, :3]
    angles = lattice[:, 3:]
    lengths = lengths / float(num_atoms) ** 1/3
    lscaled_lattice = torch.cat([lengths, angles], dim = 1)
    atomic_nums = torch.tensor(mol.get_atomic_numbers(), dtype =torch.long)
    gdict = {'lattice': lattice,
             'lscaled_lattice': lscaled_lattice,
             'atomic_nums':atomic_nums,
             'num_atoms':num_atoms}
    return g, lg, gdict


def prepare_line_dgl(g):
    lg = g.line_graph(shared=True)
    lg.apply_edges(compute_bond_cosines)
    lg.ndata.pop('r')
    lg.ndata.pop('d')
    lg.ndata.pop('u')
    # g.edata.pop('r')
    return lg


def formula_to_spread_graph(string):
    elem, weight = parse_roost(string)
    alist = []
    for i, w in enumerate(weight):
        for _ in range(int(w)):
            alist.append(elem[i])
    x = torch.zeros([len(alist), 3])
    g = dgl.remove_self_loop(dgl.knn_graph(x, len(alist)))
    # build up atom attribute tensor
    sps_features = []
    for s in alist:
        feat = list(get_node_attributes(s, atom_features='cgcnn'))
        sps_features.append(feat)
    sps_features = np.array(sps_features)
    node_features = torch.tensor(sps_features).type(
        torch.get_default_dtype()
    )
    g.ndata["atom_features"] = node_features
    g.ndata['weight'] = torch.ones(len(alist)).view(-1, 1)
    # g.ndata['weight'] = torch.tensor(weight).view(-1, 1)
    return g


def formula_to_dense_graph(string):
    elem, weight = parse_roost(string)
    x = torch.zeros((len(elem), 3))
    g = dgl.remove_self_loop(dgl.knn_graph(x, len(elem)))
    sps_features = []
    for s in elem:
        feat = list(get_node_attributes(s, atom_features='cgcnn'))
        sps_features.append(feat)
    sps_features = np.array(sps_features)
    node_features = torch.tensor(sps_features).type(
        torch.get_default_dtype()
    )
    g.ndata["atom_features"] = node_features

    weights = np.atleast_2d(weight).T / np.sum(weight)
    weights = torch.tensor(weights, dtype=torch.float32)
    g.ndata["weight"] = weights
    return g


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 dataset = cfg.dataset,
                 path = cfg.data_dir,
                 cutoff= cfg.cutoff,
                 max_neighbors=cfg.max_neighbors,
                 num_workers=cfg.workers,
                 ):
        super().__init__()
        print('Load dataset')
        self.path = path
        self.formula_graph = args.fg
        self.graphs = []
        self.line_graphs = []
        self.formula = []
        self.labels = []
        self.graph_props = []

        # Prepare dataframe
        print(f"Loading data, dataset name: {dataset}")
        df = pd.read_pickle(self.path + dataset + '.pkl')

        self.num = len(df)
        self.num_train, self.num_valid = int(self.num*cfg.train_ratio), int(self.num*cfg.valid_ratio)
        self.num_test = self.num - self.num_train - self.num_valid
        print(f"num train:{self.num_train}, num valid:{self.num_valid}, num test:{self.num_test}")

        print('Prepare graph and line graph from json dict')
        datasplit = np.array_split(df['atoms'], num_workers)

        def progress_j_to_g(df):
            gs = []
            lgs = []
            gps = []
            for d in tqdm(df):
                g, lg, gdict = json_dict_to_graph(d, cutoff, max_neighbors, pbc=True)
                gs.append(g)
                lgs.append(lg)
                gps.append(gdict)
            return gs, lgs, gps

        graphs = Parallel(n_jobs=num_workers)(delayed(progress_j_to_g)(split) for split in datasplit)
        for gs, lgs, gps in graphs:
            self.graphs += gs
            self.line_graphs += lgs
            self.graph_props+= gps

        print('Prepare graph from formula')
        datasplit = np.array_split(df['full_formula'], num_workers)

        def progress_f_to_g(df):
            if self.formula_graph == 'mask':
                return [formula_to_spread_graph(d) for d in tqdm(df)]
            elif self.formula_graph == "dense":
                return [formula_to_dense_graph(d) for d in tqdm(df)]
            else:
                assert False, "Unspecified formula graph style"

        graphs = Parallel(n_jobs=num_workers)(delayed(progress_f_to_g)(split) for split in datasplit)
        for gs in graphs:
            self.formula += gs

        print('Prepare labels')
        self.labels = df[args.target].to_numpy()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        g = self.graphs[index]
        lg = self.line_graphs[index]
        fg = self.formula[index]
        labels = self.labels[index]
        # torch.geometric data
        gdict = self.graph_props[index]
        gprops = Data(
            atomic_nums = gdict['atomic_nums'],
            num_atoms = gdict['num_atoms'],
            num_nodes = gdict['num_atoms'],
            lattice = gdict['lattice'],
            lscaled_lattice = gdict['lscaled_lattice']
        )
        return g, lg, fg, gprops, labels

    @staticmethod
    def collate(samples):
        g, lg, fg, gprops, labels = map(list, zip(*samples))
        g = dgl.batch(g)
        lg = dgl.batch(lg)
        fg = dgl.batch(fg)
        gdata = Batch.from_data_list(gprops)
        return (g, lg), fg, gdata, torch.tensor(labels, dtype =torch.float32).view(-1,1)

def get_scaler(data_list):
    lattice_tensor = torch.cat([v[-2].lscaled_lattice for v in data_list])
    lattice_scaler = get_scaler_from_data_tensor(lattice_tensor)
    return lattice_scaler


def MaterialsDataloader(args, dataset : str = cfg.dataset):

    print('Prepare train/validation/test data')
    data = LoadDataset(args, dataset= dataset)
    collate_fn = data.collate
    data_list = [d for d in data]
    train_set = data_list[:data.num_train]
    lattice_scaler = get_scaler(train_set)
    torch.save(lattice_scaler, cfg.data_dir  + cfg.dataset + "_LATTICE-SCALER.pt")
    valid_set = data_list[data.num_train:data.num_train + data.num_valid]
    test_set = data_list[data.num_train + data.num_valid:]

    train_loader = DataLoader(train_set,
                              batch_size=cfg.TRAIN.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set,
                              batch_size=cfg.TRAIN.batch_size,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_set,
                             batch_size=cfg.TRAIN.batch_size,
                             collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader

class LoadEvalset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 dataset = cfg.evalset,
                 path = cfg.data_dir,
                 num_workers =cfg.workers,
                 ):
        self.args = args
        print(f"Loading eval dataset : {dataset}")
        df = pd.read_pickle(path + dataset +".pkl")
        # target
        df = df.dropna(subset = [args.target]).reset_index(drop = True)
        # df = df[df[target] > 0].reset_index(drop = True)
        self.formula_graph = args.fg
        self.prop = args.target
        self.formula = []
        self.labels= []

        self.num = len(df)
        self.num_train, self.num_valid = int(self.num*cfg.train_ratio), int(self.num*cfg.valid_ratio)
        self.num_test = self.num - self.num_train - self.num_valid
        print(f"num data:{self.num_train}, num valid:{self.num_valid}, num test:{self.num_test}")

        print('Prepare graph from formula')
        datasplit = np.array_split(df['full_formula'], num_workers)

        def progress_f_to_g(df):
            if self.formula_graph == 'mask':
                return [formula_to_spread_graph(d) for d in tqdm(df)]
            elif self.formula_graph == "dense":
                return [formula_to_dense_graph(d) for d in tqdm(df)]
            else:
                assert False, "Unspecified formula graph style"
        graphs = Parallel(n_jobs=num_workers)(delayed(progress_f_to_g)(split) for split in datasplit)
        for gs in graphs:
            self.formula += gs
        print('Prepare labels')
        self.labels = df[args.target].to_numpy()

        if args.weight == '3d':
            self.graphs = []
            self.line_graphs = []
            print('Prepare graph and line graph from json dict')
            datasplit = np.array_split(df['atoms'], num_workers)

            def progress_j_to_g(df):
                gs = []
                lgs = []
                for d in tqdm(df):
                    g, lg,_ = json_dict_to_graph(d, cfg.cutoff, cfg.max_neighbors, pbc=True)
                    gs.append(g)
                    lgs.append(lg)
                return gs, lgs

            graphs = Parallel(n_jobs=num_workers)(delayed(progress_j_to_g)(split) for split in datasplit)
            for gs, lgs in graphs:
                self.graphs += gs
                self.line_graphs += lgs


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.args.weight == '3d':
            g = self.graphs[index]
            lg = self.line_graphs[index]
            fg = self.formula[index]
            labels = self.labels[index]
            return g, lg, fg, labels
        fg = self.formula[index]
        labels = self.labels[index]
        return fg, labels

    @staticmethod
    def collate(samples):
        fg, labels = map(list, zip(*samples))
        fg = dgl.batch(fg)
        return fg, torch.tensor(labels, dtype=torch.float32).view(-1,1)
    @staticmethod
    def collate_3d(samples):
        g, lg, fg, labels = map(list, zip(*samples))
        g = dgl.batch(g)
        lg = dgl.batch(lg)
        fg = dgl.batch(fg)
        return (g, lg), fg, torch.tensor(labels, dtype=torch.float32).view(-1, 1)

def EvalLoader(args, dataset = cfg.evalset):
    data = LoadEvalset(args, dataset=dataset)
    if args.weight == '3d':
        collate_fn = data.collate_3d
    else:
        collate_fn = data.collate
    data_list = [d for d in data]
    train_set = data_list[:data.num_train]
    valid_set = data_list[data.num_train:data.num_train+data.num_valid]
    test_set = data_list[data.num_train+data.num_valid:]

    train_loader = DataLoader(train_set,
                              batch_size=cfg.EVAL.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set,
                              batch_size=cfg.EVAL.batch_size,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_set,
                             batch_size=cfg.EVAL.batch_size,
                             collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader


class Node(object):
    """ Node class for tree data structure """

    def __init__(self, parent, val=None):
        self.value = val
        self.parent = parent
        self.children = []

    def __repr__(self):
        return f"<Node {self.value} >"


def format_composition(comp):
    """ format str to ensure weights are explicate
    example: BaCu3 -> Ba1Cu3
    """
    subst = r"\g<1>1.0"
    comp = re.sub(r"[\d.]+", lambda x: str(float(x.group())), comp.rstrip())
    comp = re.sub(r"([A-Z][a-z](?![0-9]))", subst, comp)
    comp = re.sub(r"([A-Z](?![0-9]|[a-z]))", subst, comp)
    comp = re.sub(r"([\)](?=[A-Z]))", subst, comp)
    comp = re.sub(r"([\)](?=\())", subst, comp)
    return comp


def parenthetic_contents(string):
    """
    Generate parenthesized contents in string as (level, contents, weight).
    """
    num_after_bracket = r"[^0-9.]"

    stack = []
    for i, c in enumerate(string):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            start = stack.pop()
            num = re.split(num_after_bracket, string[i + 1:])[0] or 1
            yield {
                "value": [string[start + 1: i], float(num), False],
                "level": len(stack) + 1,
            }

    yield {"value": [string, 1, False], "level": 0}


def build_tree(root, data):
    """ build a tree from ordered levelled data """
    for record in data:
        last = root
        for _ in range(record["level"]):
            last = last.children[-1]
        last.children.append(Node(last, record["value"]))


def update_weights(comp, weight):
    """ split composition string into elements (keys) and weights
    example: Ba1Cu3 -> [Ba,Cu] [1,3]
    """
    regex3 = r"(\d+\.\d+)|(\d+)"
    parsed = [j for j in re.split(regex3, comp) if j]
    elements = parsed[0::2]
    weights = [float(p) * weight for p in parsed[1::2]]
    new_comp = ""
    for m, n in zip(elements, weights):
        new_comp += m + f"{n:.2f}"
    return new_comp


def update_parent(child):
    """ update the str for parent """
    input_str = child.value[2] or child.value[0]
    new_str = update_weights(input_str, child.value[1])
    pattern = re.escape("(" + child.value[0] + ")" + str(child.value[1]))
    old_str = child.parent.value[2] or child.parent.value[0]
    child.parent.value[2] = re.sub(pattern, new_str, old_str, 0)


def reduce_tree(current):
    """ perform a post-order reduction on the tree """
    if not current:
        pass

    for child in current.children:
        reduce_tree(child)
        update_parent(child)


def splitout_weights(comp):
    """ split composition string into elements (keys) and weights
    example: Ba1Cu3 -> [Ba,Cu] [1,3]
    """
    elements = []
    weights = []
    regex3 = r"(\d+\.\d+)|(\d+)"
    try:
        parsed = [j for j in re.split(regex3, comp) if j]
    except:
        print("parsed:", comp)
    elements += parsed[0::2]
    weights += parsed[1::2]
    weights = [float(w) for w in weights]
    return elements, weights


def parse_roost(string):
    # format the string to remove edge cases
    string = format_composition(string)
    # get nested bracket structure
    nested_levels = list(parenthetic_contents(string))
    if len(nested_levels) > 1:
        # reverse nested list
        nested_levels = nested_levels[::-1]
        # plant and grow the tree
        root = Node("root", ["None"] * 3)
        build_tree(root, nested_levels)
        # reduce the tree to get compositions
        reduce_tree(root)
        return splitout_weights(root.children[0].value[2])

    else:
        return splitout_weights(string)
