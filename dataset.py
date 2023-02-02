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
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from p_tqdm import p_umap
from tqdm import tqdm
from utils import *
from jarvis.core.specie import get_node_attributes
from jarvis.core.specie import Specie
from joblib import Parallel, delayed
from config import cfg

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
def formula_to_graph(string):
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
    for i, _ in enumerate(alist):
        self_idx += [i] * len(alist)
        nbr_idx += list(range(len(alist)))
        atom_types.append(Specie(alist[i]).Z)
    edge_idx = np.concatenate((self_idx, nbr_idx)).reshape(2,-1)
    node_features = []
    for s in alist:
        feat = list(get_node_attributes(s, atom_features='cgcnn'))
        node_features.append(feat)
    node_features = np.array(node_features)
    # elem_weights = np.atleast_2d(weight) / np.sum(weight)
    elem_weights = np.ones(len(alist)).reshape(-1,1)
    atom_types = np.array(atom_types)
    return atom_types, node_features, edge_idx, elem_weights

def formula_to_dense_graph(string):
    elem, weight = parse_roost(string)
    # build up atom attribute tensor
    self_idx = []
    nbr_idx = []
    atom_types = []
    node_features = []
    for i, s in enumerate(elem):
        self_idx += [i] * len(elem)
        nbr_idx += list(range(len(elem)))
        atom_types.append(Specie(elem[i]).Z)
        feat = list(get_node_attributes(s, atom_features='cgcnn'))
        node_features.append(feat)
    edge_idx = np.concatenate((self_idx, nbr_idx)).reshape(2,-1)
    node_features = np.array(node_features)
    w_attr = []
    for i in range(len(self_idx)):
        w_attr.append(weight[nbr_idx[i]]/(weight[self_idx[i]]+ weight[nbr_idx[i]]))
    elem_weights = np.atleast_2d(weight) / np.sum(weight)
    w_attr = np.array(w_attr).reshape(-1,1)
    atom_types = np.array(atom_types)
    return atom_types, node_features, edge_idx, w_attr #w_attr

def process_one(mp_ids, crystal_array, crystal_string, props):
    crystal = build_crystal(crystal_array)
    graph_arrays = build_crystal_graph(crystal, 'crystalnn')
    formula_arrays = formula_to_dense_graph(crystal_string)
    properties = {'prop': np.array(props)}
    result_dict = {
        'mp_ids': mp_ids,
        'graph_arrays': graph_arrays,
        'formula_arrays':formula_arrays,
    }
    result_dict.update(properties)
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

class PairData(Data):
    def __init__(self, edge_index_s =None, atom_types_s = None,
                 edge_index_t = None, atom_types_t = None, **kwargs):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.atom_types_s = atom_types_s
        self.edge_index_t = edge_index_t
        self.atom_types_t = atom_types_t
        for key, value in kwargs.items():
            setattr(self, key, value)
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.atom_types_s.size(0)
        if key == 'edge_index_t':
            return self.atom_types_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num, max_atoms, num_workers=12, path = None):
        super().__init__()
        print('Load dataset')
        self.path = path
        self.cached_data = []
        self.prop = cfg.PROP
        random.seed(123)
        # Prepare dataframe(jarvis jdata to dataframe)
        if self.path != None:
            df = pd.read_pickle(self.path + dataset + '.pkl')
        else:
            df = curate(dataset, num, max_atoms)
            df.to_pickle("./data/" + dataset + ".pkl")

        datasplit = np.array_split(df, num_workers)

        def progress(df):
            return [process_one(mp_ids, crystal_array, crystal_string, props) for (mp_ids, crystal_array, crystal_string, props) in zip(df['id'], tqdm(df['atoms']), df['full_formula'], df[self.prop])]

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
        # prop = self.scaler.transform(data_dict['prop'])
        prop = torch.Tensor(data_dict['prop'])
        data = PairData(
            node_features_s = torch.Tensor(f_node_features),
            atom_types_s=torch.LongTensor(f_atom_types),
            edge_index_s=torch.LongTensor(
                f_edge_indices),  # shape (2, num_edges)
            num_atoms_s=f_atom_types.shape[0],
            num_bonds_s=f_edge_indices.shape[-1],
            atom_weights_s = torch.Tensor(f_elem_weights).reshape(-1,1),
            frac_coords_t=torch.Tensor(frac_coords),
            atom_types_t=torch.LongTensor(atom_types),
            lengths_t=torch.Tensor(lengths).view(1, -1),
            angles_t=torch.Tensor(angles).view(1, -1),
            edge_index_t=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages_t=torch.LongTensor(to_jimages),
            num_atoms_t=num_atoms,
            num_bonds_t=edge_indices.shape[0],
            y = prop.view(1,-1)
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

def MaterialLoader():
    num = cfg.NUM_TRAIN + cfg.NUM_VALID + cfg.NUM_TEST
    data = LoadDataset(cfg.DATASET_NAME, num, cfg.MAX_ATOMS, path = cfg.DATA_DIR)

    train_cash = data.cached_data[:cfg.NUM_TRAIN]
    lattice_scaler = get_scaler(train_cash)
    torch.save(lattice_scaler, cfg.SCALER_DIR + "lattice_scaler.pt")
    data_list = [d for d in data]
    train_set = data_list[:cfg.NUM_TRAIN]
    valid_set = data_list[cfg.NUM_TRAIN:cfg.NUM_TRAIN+cfg.NUM_VALID]
    test_set = data_list[cfg.NUM_TRAIN+cfg.NUM_VALID:]

    train_loader = DataLoader(train_set,
                              batch_size=cfg.TRAIN.batch_size,
                              shuffle=True,
                              follow_batch=['atom_types_s', 'atom_types_t'])
    valid_loader = DataLoader(valid_set,
                              batch_size=cfg.TRAIN.batch_size,
                              shuffle=True,
                              follow_batch=['atom_types_s', 'atom_types_t'])
    test_loader = DataLoader(test_set,
                             batch_size=cfg.TRAIN.batch_size,
                             shuffle=True,
                             follow_batch = ['atom_types_s', 'atom_types_t'])
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = MaterialLoader()
    print("")