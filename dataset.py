from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from jarvis.db.figshare import data as jdata
import numpy as np
import pandas as pd
import random
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from p_tqdm import p_umap
from tqdm import tqdm
from utils import *
from jarvis.core.specie import get_node_attributes
from jarvis.core.specie import Specie
from joblib import Parallel, delayed

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)

def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = max(min(val, 1), -1)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])

def build_crystal(atoms):
    '''
    build crystal from strings, atoms is a dict with 'lattice_mat', elements' as keys
    convert jarvis data to pymatgen structure data
    structure can be build from lattice parameters or from files
    "https://pymatgen.org/usage.html#"
    '''
    crystal = Structure(atoms['lattice_mat'], atoms['elements'], atoms['coords'], coords_are_cartesian= False)
    crystal = crystal.get_reduced_structure()
    canonical_crystal = Structure(
        lattice = Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords = crystal.frac_coords,
        coords_are_cartesian=False
    )
    return canonical_crystal

def build_crystal_graph(crystal, graph_method='crystalnn'):
    """
    build graph array from pymatgen structure data
    """

    if graph_method == 'crystalnn':
        crystal_graph = StructureGraph.with_local_env_strategy(
            crystal, CrystalNN)
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix,
                       lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms

# def preprocess_tensors(crystal_array_list, ids, graph_method):
#     def process_one(batch_idx, crystal_array, id, graph_method):
#         crystal = build_crystal(crystal_array)
#         graph_arrays = build_crystal_graph(crystal, graph_method)
#         result_dict = {
#             'batch_idx': batch_idx,
#             'mp_id': id,
#             'graph_arrays': graph_arrays,
#         }
#         return result_dict
#
#     unordered_results = p_umap(
#         process_one,
#         list(range(len(crystal_array_list))),
#         crystal_array_list, ids,
#         [graph_method] * len(crystal_array_list),
#         num_cpus=12,
#     )
#     ordered_results = list(
#         sorted(unordered_results, key=lambda x: x['batch_idx']))
#     return ordered_results

def add_scaled_lattice_prop(data_list, lattice_scale_method):
    for dict in data_list:
        graph_arrays = dict['graph_arrays']
        # the indexes are brittle if more objects are returned
        lengths = graph_arrays[2]
        angles = graph_arrays[3]
        num_atoms = graph_arrays[-1]
        assert lengths.shape[0] == angles.shape[0] == 3
        assert isinstance(num_atoms, int)

        if lattice_scale_method == 'scale_length':
            lengths = lengths / float(num_atoms)**(1/3)

        dict['scaled_lattice'] = np.concatenate([lengths, angles])
# spread formula#############################
def formula_to_spread_graph(string):
    elem, weight = parse_roost(string)
    alist, wlist = [], []
    for i, w in enumerate(weight):
        for _ in range(int(w)):
            alist.append(elem[i])
            wlist.append(weight[i])
    elem_w =[v/sum(wlist) for v in wlist]
    # build up atom attribute tensor
    self_idx = []
    nbr_idx = []
    atom_types = []
    w_attr = []
    for i, _ in enumerate(alist):
        self_idx += [i] * len(alist)
        nbr_idx += list(range(len(alist)))
        w_attr+= [elem_w[v] for v in range(len(alist))]
        atom_types.append(Specie(alist[i]).Z)
    edge_idx = np.concatenate((self_idx, nbr_idx)).reshape(2,-1)
    w_attr = np.array(w_attr).reshape(-1, 1)
    node_features = []
    for s in alist:
        feat = list(get_node_attributes(s, atom_features='cgcnn'))
        node_features.append(feat)
    node_features = np.array(node_features)
    # elem_weights = np.atleast_2d(weight) / np.sum(weight)
    atom_types = np.array(atom_types)
    return atom_types, node_features, edge_idx, w_attr

def formula_to_dense_graph(string):
    elem, weight = parse_roost(string)
    alist = []
    for i, w in enumerate(weight):
        for _ in range(int(w)):
            alist.append(elem[i])
    x = torch.zeros([len(alist),3])
    # build up atom attribute tensor
    self_idx = []
    nbr_idx = []
    atom_types = []
    for i, _ in enumerate(elem):
        self_idx += [i] * len(elem)
        nbr_idx += list(range(len(elem)))
        atom_types.append(Specie(elem[i]).Z)
    edge_idx = np.concatenate((self_idx, nbr_idx)).reshape(2,-1)
    node_features = []
    for s in elem:
        feat = list(get_node_attributes(s, atom_features='cgcnn'))
        node_features.append(feat)
    node_features = np.array(node_features)
    elem_weights = np.atleast_2d(weight) / np.sum(weight)
    atom_types = np.array(atom_types)
    return atom_types, node_features, edge_idx, elem_weights

def process_one(mp_ids, crystal_array, crystal_string, labels):
    crystal = build_crystal(crystal_array)
    graph_arrays = build_crystal_graph(crystal, 'crystalnn')
    formula_arrays = formula_to_dense_graph(crystal_string)
    label_arrays = np.array(labels)
    result_dict = {
        'mp_ids': mp_ids,
        'graph_arrays': graph_arrays,
        'formula_arrays':formula_arrays,
        'label_arrays': label_arrays
    }
    return result_dict

def curate(dataset, num, max_atoms):
    def refine(df):
        short_inx = []
        for i in df.index:
            d = df.loc[i]['atoms']['elements']
            if len(d) < max_atoms:
                short_inx.append(i)
        df = df.loc[short_inx]
        mask = df['full_formula'].duplicated()
        un_dup_inx = []
        if mask.sum() > 0:
            duplicated_formula_set = set(df['full_formula'][mask].to_list())
            for f in duplicated_formula_set:
                fenergy = df['formation_energy_per_atom'][df['full_formula'] == f]
                un_dup_inx.append(fenergy.index[fenergy.argmin()])
        ainx = df[~mask].index.to_list() + un_dup_inx
        return df.loc[ainx].reset_index(drop=True)
    num_buff = int(num*1.5)
    df = pd.DataFrame(jdata(dataset))[:num_buff]
    df_ = refine(df)
    cnt = 0
    while len(df_) < num:
        cnt+=1
        print(f"{cnt} curating dataset\n")
        num_diff = num - len(df_)
        df = pd.concat([df_, pd.DataFrame(jdata(dataset))[num_buff:num_buff+num_diff*2]])
        df_ = refine(df)
        num_buff = num_buff+num_diff*2
    if len(df_) > num:
        df_ = df_.iloc[:num]
    return df_


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num, max_atoms, num_workers=12, path = None):
        super().__init__()
        print('Load dataset')
        self.path = path
        self.cached_data = []
        random.seed(123)
        # Prepare dataframe(jarvis jdata to dataframe)
        if self.path != None:
            df = pd.read_pickle(self.path + dataset + '.pkl')
        else:
            df = curate(dataset, num, max_atoms)
            df.to_pickle("./data/" + dataset + ".pkl")

        datasplit = np.array_split(df, num_workers)

        def progress(df):
            return [process_one(mp_ids, crystal_array, crystal_string, labels) for (mp_ids, crystal_array, crystal_string,labels) in zip(df['id'], tqdm(df['atoms']), df['full_formula'], df['formation_energy_per_atom'])]

        cached_data = Parallel(n_jobs=num_workers)(delayed(progress)(split) for split in datasplit)

        for data in cached_data:
            self.cached_data+=data

        # self.cached_data = preprocess_tensors(
        #     crystal_array_list,mp_ids,
        #     graph_method='crystalnn')
        #
        add_scaled_lattice_prop(self.cached_data, lattice_scale_method = 'scale_length')
        # self.lattice_scaler = None
        # self.scaler = None
    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']
        (f_atom_types, f_node_features,
         f_edge_indices, f_elem_weights) = data_dict['formula_arrays']
        y = data_dict['label_arrays']
        data = Data(
            node_features = torch.Tensor(f_node_features),
            atom_types=torch.LongTensor(f_atom_types),
            edge_index=torch.LongTensor(
                f_edge_indices),  # shape (2, num_edges)
            num_atoms=f_atom_types.shape[0],
            num_bonds=f_edge_indices.shape[-1],
            atom_weights = torch.Tensor(f_elem_weights).reshape(-1,1),
            num_nodes=f_atom_types.shape[0],
            y = torch.Tensor(y),
        )
    #     data = Data(
    #         frac_coords=torch.Tensor(frac_coords),
    #         atom_types=torch.LongTensor(atom_types),
    #         lengths=torch.Tensor(lengths).view(1, -1),
    #         angles=torch.Tensor(angles).view(1, -1),
    #         edge_index=torch.LongTensor(
    #             edge_indices.T).contiguous(),  # shape (2, num_edges)
    #         to_jimages=torch.LongTensor(to_jimages),
    #         num_atoms=num_atoms,
    #         num_bonds=edge_indices.shape[0],
    #         num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
    # )
        data.update(lengths=torch.Tensor(lengths).view(1, -1),
                    angles=torch.Tensor(angles).view(1, -1),
                    target_coords = torch.Tensor(frac_coords),
                    target_atom_types = torch.LongTensor(atom_types),
                    target_edge_index = torch.LongTensor(
                        edge_indices.T).contiguous(),
                    to_jimages = torch.LongTensor(to_jimages),
                    target_num_atoms = atom_types.shape[0]
                    )
        return data



    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"

def get_scaler(data_list):
    # Load once to compute property scaler
    lattice_scaler = get_scaler_from_data_list(data_list,key='scaled_lattice')
    # train_dataset.scaler = get_scaler_from_data_list(
    #     train_dataset.cached_data,
    #     key=train_dataset.prop)
    return lattice_scaler

def MaterialLoader(args):
    num = args.num_train + args.num_valid + args.num_test
    data = LoadDataset(args.dataset, num, args.max_atoms, path = args.data_path)

    train_cash = data.cached_data[:args.num_train]
    lattice_scaler = get_scaler(train_cash)
    torch.save(lattice_scaler, "./data/lattice_scaler.pt")
    data_list = [d for d in data]
    train_set = data_list[:args.num_train]
    valid_set = data_list[args.num_train:args.num_train+args.num_valid]
    test_set = data_list[args.num_train+args.num_valid:]

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=True)
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=' prediction training')
    parser.add_argument('--layers', type=int, default=4, help="")
    parser.add_argument('--atom-input-features', type=int, default=92, help="")
    parser.add_argument('--hidden-features', type=int, default=256, help="")
    parser.add_argument('--output-features', type=int, default=9, help="")
    parser.add_argument('--n-heads', type=int, default=4, help="")
    parser.add_argument('--dataset', type = str, default='mp_3d_2020')
    parser.add_argument('--num-train', type=int, default=100, help="")
    parser.add_argument('--num-valid', type=int, default=50, help="")
    parser.add_argument('--num-test', type=int, default=50, help="")
    parser.add_argument('--batch-size', type=int, default=10, help="")
    parser.add_argument('--data-path', type = str, default="./data/")
    args = parser.parse_args()
    train_loader, valid_loader, test_loader = MaterialLoader(args)
    print("")