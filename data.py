from torch.utils.data import DataLoader
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



def formula_to_graph(string):
    elem, weight = parse_roost(string)
    alist = []
    for i, w in enumerate(weight):
        for _ in range(int(w)):
            alist.append(elem[i])
    x = torch.zeros([len(alist),3])
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
    return g



class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num, target = [], num_workers=7):
        super().__init__()
        print('Load dataset')
        self.graphs = []
        self.labels = []

        # Prepare dataframe
        df = pd.DataFrame(jdata(dataset))[:num]
        print('Prepare labels')
        datasplit = np.array_split(df['atoms'], num_workers)

        def get_lattice_mat(atom_dict):
            return np.array(atom_dict['lattice_mat']).reshape(-1)
        def progress_label(df):
            return [get_lattice_mat(d) for d in tqdm(df)]
        labels = Parallel(n_jobs=num_workers)(delayed(progress_label)(split) for split in datasplit)
        for lbls in labels:
            self.labels += lbls

        print('Prepare formula graph')
        datasplit = np.array_split(df['full_formula'], num_workers)
        def progress_f_to_g(df):
            return [formula_to_graph(d) for d in tqdm(df)]
        graphs = Parallel(n_jobs=num_workers)(delayed(progress_f_to_g)(split) for split in datasplit)
        for gs in graphs:
            self.graphs += gs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        fg = self.graphs[index]
        labels = self.labels[index]
        return fg, labels

    @staticmethod
    def collate(samples):
        fg, labels = map(list, zip(*samples))
        fg = dgl.batch(fg)
        return fg, torch.tensor(labels)



def MaterialsDataloader(args):
    '''
    Materials dataloader.
    :param args: arguments
    :return: graph dataloader
    '''

    num = args.num_train + args.num_valid + args.num_test
    idx = list(range(num))
    random.seed(123)
    random.shuffle(idx)
    train_indices = idx[0:args.num_train]
    valid_indices = idx[args.num_train:args.num_train + args.num_valid]
    test_indices = idx[args.num_train + args.num_valid:args.num_train + args.num_valid + args.num_test]

    print('Prepare train/validation/test data')
    data = LoadDataset(args.dataset, num, target=args.target,
                       num_workers=args.num_workers)
    collate_fn = data.collate

    train_data = torch.utils.data.Subset(data, train_indices)
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              #   num_workers=args.num_workers,
                              collate_fn=collate_fn)
    valid_data = torch.utils.data.Subset(data, valid_indices)
    valid_loader = DataLoader(valid_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              # num_workers=args.num_workers,
                              collate_fn=collate_fn)
    test_data = torch.utils.data.Subset(data, test_indices)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             # num_workers=args.num_workers,
                             collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader, test_indices

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