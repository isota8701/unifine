import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
import dgl
import numpy as np
import pandas as pd
import random, math
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm
from utils import *
from jarvis.core.specie import get_node_attributes
from jarvis.core.specie import Specie
from joblib import Parallel, delayed
from config import cfg, cfg_from_file

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
    #oqmd dataset error
    atoms['elements'] = [v.strip() for v in atoms['elements']]
    try:
        crystal = Structure(atoms['lattice_mat'],
                        atoms['elements'],
                        atoms['coords'], coords_are_cartesian=False)
    except:
        return None
    sga = SpacegroupAnalyzer(crystal)
    spacegroup = get_spacegroup(sga.get_crystal_system())
    spacegroup_no = sga.get_space_group_number() - 1

    crystal = crystal.get_reduced_structure()
    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False
    )
    sps_features = []
    mol = Atoms.from_dict(atoms).ase_converter()
    for s in mol.get_chemical_symbols():
        feat = list(get_node_attributes(s, atom_features='cgcnn'))
        sps_features.append(feat)
    sps_features = np.array(sps_features)
    return canonical_crystal, (spacegroup, spacegroup_no), sps_features


def build_crystal_graph(crystal, atoms, graph_method='crystalnn'):
    """
    build graph array from pymatgen structure data
    """
    if graph_method == 'crystalnn':
        try:
            crystal_graph = StructureGraph.with_local_env_strategy(
            crystal, CrystalNN)
        except:
            print(f'Voronoi Neighbors Issue, formula {crystal.formula}')
            try:
                crystal.perturb(0.001)
                crystal_graph = StructureGraph.with_local_env_strategy(crystal, CrystalNN)
            except:
                return None

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


def get_spacegroup(crystal_system):
    if crystal_system == "triclinic":
        return 0
    elif crystal_system == "monoclinic":
        return 1
    elif crystal_system == "orthorhombic":
        return 2
    elif crystal_system == "tetragonal":
        return 3
    elif crystal_system == "trigonal":
        return 4
    elif crystal_system == "hexagonal":
        return 5
    elif crystal_system == "cubic":
        return 6


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
            lengths = lengths / float(num_atoms) ** (1 / 3)

        dict['scaled_lattice'] = np.concatenate([lengths, angles])


# spread formula#############################
def formula_to_spread_graph(string):
    elem, weight = parse_roost(string)
    if 1.01>= sum(weight) > 0.99:
        '''
        oqmd stability data need further modify!!!!!!!!!!!!!!!
        evaluate dataset oqmd weight = [0.25, 0.25 ...]
        [0.33333, 0.33333 ...]
        [0.4,0.6]
        '''
        mm = min(weight)
        if max([w % mm for w in weight]) > 0.001:
            weight = [round(w/mm*10) for w in weight]
            mm = math.gcd(*weight)
        weight = [round(w/mm) for w in weight]
        if max(weight) > cfg.MAX_ATOMS:
            return None

    alist = []
    for i, w in enumerate(weight):
        for _ in range(int(w)):
            alist.append(elem[i])
    x = torch.zeros([len(alist), 3])
    # build up atom attribute tensor
    self_idx = []
    nbr_idx = []
    atom_types = []
    for i, _ in enumerate(alist):
        self_idx += [i] * len(alist)
        nbr_idx += list(range(len(alist)))
        atom_types.append(Specie(alist[i]).Z)
    edge_idx = np.concatenate((self_idx, nbr_idx)).reshape(2, -1)
    node_features = []
    for s in alist:
        feat = list(get_node_attributes(s, atom_features='cgcnn'))
        node_features.append(feat)
    node_features = np.array(node_features)
    # elem_weights = np.atleast_2d(weight) / np.sum(weight)
    elem_weights = np.ones(shape=(edge_idx.shape[1], 1))
    # elem_weights = np.ones(len(alist)).reshape(-1,1) # roost style?
    atom_types = np.array(atom_types)
    return atom_types, node_features, edge_idx, elem_weights

def atoms_to_graph(atom):
    elems = atom['elements']
    elem_dict = {key: elems.count(key) for key in np.unique(elems)}
    elem, weight = list(elem_dict.keys()), list(elem_dict.values())
    alist = []
    for i, w in enumerate(weight):
        for _ in range(int(w)):
            alist.append(elem[i])
    x = torch.zeros([len(alist), 3])
    # build up atom attribute tensor
    self_idx = []
    nbr_idx = []
    atom_types = []
    for i, _ in enumerate(alist):
        self_idx += [i] * len(alist)
        nbr_idx += list(range(len(alist)))
        atom_types.append(Specie(alist[i]).Z)
    edge_idx = np.concatenate((self_idx, nbr_idx)).reshape(2, -1)
    node_features = []
    for s in alist:
        feat = list(get_node_attributes(s, atom_features='cgcnn'))
        node_features.append(feat)
    node_features = np.array(node_features)
    # elem_weights = np.atleast_2d(weight) / np.sum(weight)
    elem_weights = np.ones(shape=(edge_idx.shape[1], 1))
    # elem_weights = np.ones(len(alist)).reshape(-1,1) # roost style?
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
    edge_idx = np.concatenate((self_idx, nbr_idx)).reshape(2, -1)
    node_features = np.array(node_features)
    w_attr = []
    for i in range(len(self_idx)):
        w_attr.append(weight[nbr_idx[i]] / (weight[self_idx[i]] + weight[nbr_idx[i]]))
    elem_weights = np.atleast_2d(weight) / np.sum(weight)
    w_attr = np.array(w_attr).reshape(-1, 1)
    atom_types = np.array(atom_types)
    return atom_types, node_features, edge_idx, w_attr  # w_attr


def process_one(mp_ids, crystal_array, crystal_string, props):
    crystal_info = build_crystal(crystal_array)
    if crystal_info is None:
        result_dict = {'graph_issue': True}
        return result_dict
    else:
        crystal, (spacegroup, spacegroup_no), atom_feats = crystal_info
    graph_arrays = build_crystal_graph(crystal, crystal_array, 'crystalnn')
    if graph_arrays is None:
        result_dict = {'graph_issue': True}
        return result_dict
    if cfg.DATASET_NAME.split('_')[0] =='oqmd':
        formula_arrays = atoms_to_graph(crystal_array)
    else:
        formula_arrays = formula_to_spread_graph(crystal_string)

    properties = {'prop': np.array(props)}
    # spacegroups = {'spacegroups': (np.array(spacegroup), np.array(spacegroup_no))}
    result_dict = {
        'mp_ids': mp_ids,
        'graph_arrays': graph_arrays,
        'formula_arrays': formula_arrays,
        'spacegroups': (spacegroup, spacegroup_no),
        'atom_feats': atom_feats,
    }
    result_dict.update(properties)
    # result_dict.update(spacegroups)
    return result_dict


def curate(dataset, max_atoms):
    def refine(df):
        formula, formationE = 'full_formula','formation_energy_per_atom'
        df = df[df['atoms'].apply(lambda x: len(x['elements']) < max_atoms)]
        # df['_formula'] = df['atoms'].apply(lambda x: x['elements'])
        d1 = df[df[formula].duplicated(keep=False) == False]
        d2 = df[df[formula].duplicated(keep=False)]
        min_fe = d2.groupby(formula).min(formationE)[formationE].values
        d2 = d2[d2[formationE].apply(lambda x: any(x == min_fe))]
        df_ = pd.concat([d1, d2])
        df_ = df_.sample(frac=1).reset_index(drop=True)
        return df_
    def refine_oqmd(df):
        df = df[df['atoms'].apply(lambda x: len(x['elements']) < max_atoms)]
        dd = df.copy()
        def to_formula(alist):
            res = ''
            for e in np.unique(alist):
                res+=e.strip()
                res+=str(alist.count(e))
            return res
        dd['_formula'] = df['atoms'].apply(lambda x: to_formula(x['elements']))
        d1 = dd[dd['_formula'].duplicated(keep=False) == False]
        d2 = dd[dd['_formula'].duplicated(keep=False)]
        min_fe = d2.groupby('_formula').min('_oqmd_stability')['_oqmd_stability'].values
        d2 = d2[d2['_oqmd_stability'].apply(lambda x: any(x == min_fe))]
        df_ = pd.concat([d1, d2])
        df_ = df_.sample(frac=1).reset_index(drop=True)
        return df_

    df = pd.DataFrame(jdata(dataset))
    if cfg.DATASET_NAME.split('_')[0] == 'oqmd':
        df = refine_oqmd(df)
    else:
        df = refine(df)
    print(f'Refined dataset shape {df.shape}')
    new_num = len(df)
    new_train, new_valid = int(new_num * cfg.TRAIN_RATIO), int(new_num * cfg.VALID_RATIO)
    new_test = new_num - new_train - new_valid
    return df, new_train, new_valid, new_test


class PairData(Data):
    def __init__(self, edge_index_s=None, atom_types_s=None,
                 edge_index_t=None, atom_types_t=None, **kwargs):
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
    def __init__(self, dataset, max_atoms, num_workers=12):
        super().__init__()
        print('Load dataset')
        self.path = cfg.DATA_DIR
        from_path = cfg.FROM_PATH
        cut_data = cfg.CUT_DATA
        self.cached_data = []
        self.prop = cfg.PROP
        random.seed(cfg.RANDOM_SEED)
        # Prepare dataframe(jarvis jdata to dataframe)
        if from_path:
            print(f'Loading from path, dataset name : {dataset}')
            df = pd.read_pickle(self.path + dataset + '.pkl')
            if cut_data:
                self.num_train, self.num_valid, self.num_test = cfg.NUM_TRAIN, cfg.NUM_VALID, cfg.NUM_TEST
                self.num = self.num_train + self.num_valid + self.num_test
                assert len(df) > self.num, "less than query number"
                df = df.iloc[:self.num]
                df.to_pickle(self.path + dataset + f"_CUT_{self.num}"+".pkl")
            else:
                self.num = len(df)
                self.num_train, self.num_valid = int(self.num * cfg.TRAIN_RATIO), int(self.num * cfg.VALID_RATIO)
                self.num_test = self.num - self.num_train - self.num_valid

            print(f"NUM TRAIN:{self.num_train}, NUM VALID:{self.num_valid}, NUM_TEST:{self.num_test}")

        else:
            print("Init, call json file and refine")
            df, self.num_train, self.num_valid, self.num_test = curate(dataset, max_atoms)
            self.num = self.num_train + self.num_valid + self.num_test
            df.to_pickle("../data/" + dataset + f"_maxA{max_atoms}_N" + str(self.num) + ".pkl")
            print(f"NUM TRAIN:{self.num_train}, NUM VALID:{self.num_valid}, NUM_TEST:{self.num_test}")
        datasplit = np.array_split(df, num_workers)

        if cfg.DATASET_NAME.split('_')[0] == 'oqmd':
            def progress(df):
                return [process_one(mp_ids, crystal_array, crystal_string, props) for
                    (mp_ids, crystal_array, crystal_string, props) in
                    zip(list(df.index), tqdm(df['atoms']), df['_formula'], df[self.prop])]
        else:
            def progress(df):
                return [process_one(mp_ids, crystal_array, crystal_string, props) for
                    (mp_ids, crystal_array, crystal_string, props) in
                    zip(df['id'], tqdm(df['atoms']), df['full_formula'], df[self.prop])]


        cached_data = Parallel(n_jobs=num_workers)(delayed(progress)(split) for split in datasplit)

        for data in cached_data:
            self.cached_data += data

        cnt = 0
        for inx,data in enumerate(self.cached_data):
            if 'graph_issue' in data:
                del self.cached_data[inx]
                cnt+=1
                self.num_train-=1
                self.num-=1
        print(f'graph building issue, {cnt} discarded from train data')

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method='scale_length')
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
        (spacegroup, spacegroup_no) = data_dict['spacegroups']
        atom_feats = data_dict['atom_feats']
        # prop = self.scaler.transform(data_dict['prop'])
        prop = torch.Tensor(data_dict['prop'])
        data = PairData(
            node_features_s=torch.Tensor(f_node_features),
            atom_types_s=torch.LongTensor(f_atom_types),
            edge_index_s=torch.LongTensor(
                f_edge_indices),  # shape (2, num_edges)
            num_atoms_s=f_atom_types.shape[0],
            num_bonds_s=f_edge_indices.shape[-1],
            atom_weights_s=torch.Tensor(f_elem_weights).reshape(-1, 1),

            node_features_t=torch.Tensor(atom_feats).type(torch.get_default_dtype()),
            spacegroup_t=torch.LongTensor([spacegroup]),
            spacegroup_no_t=torch.LongTensor([spacegroup_no]),
            frac_coords_t=torch.Tensor(frac_coords),
            atom_types_t=torch.LongTensor(atom_types),
            lengths_t=torch.Tensor(lengths).view(1, -1),
            angles_t=torch.Tensor(angles).view(1, -1),
            edge_index_t=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages_t=torch.LongTensor(to_jimages),
            num_atoms_t=num_atoms,
            num_bonds_t=edge_indices.shape[0],
            y_t=prop.view(1, -1)
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


def get_scaler(data_list):
    # Load once to compute property scaler
    lattice_scaler = get_scaler_from_data_list(data_list, key='scaled_lattice')
    # train_dataset.scaler = get_scaler_from_data_list(
    #     train_dataset.cached_data,
    #     key=train_dataset.prop)
    return lattice_scaler


def MaterialLoader():
    data = LoadDataset(cfg.DATASET_NAME, cfg.MAX_ATOMS)

    train_cash = data.cached_data[:data.num_train]
    lattice_scaler = get_scaler(train_cash)
    torch.save(lattice_scaler, cfg.SCALER_DIR + "lattice_scaler.pt")
    data_list = [d for d in data]
    train_set = data_list[:data.num_train]
    valid_set = data_list[data.num_train:data.num_train + data.num_valid]
    test_set = data_list[data.num_train + data.num_valid:]

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
                             follow_batch=['atom_types_s', 'atom_types_t'])
    return train_loader, valid_loader, test_loader


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, prop, num_workers=12):
        print(f'Loading Eval dataset: {dataset}')
        path = cfg.DATA_DIR
        df = pd.read_csv(path + dataset)
        self.prop = prop
        self.cached_data = []

        self.num = len(df)
        self.num_train, self.num_valid = int(self.num * cfg.TRAIN_RATIO), int(self.num * cfg.VALID_RATIO)
        self.num_test = self.num - self.num_train - self.num_valid
        print(f"NUM TRAIN:{self.num_train}, NUM VALID:{self.num_valid}, NUM_TEST:{self.num_test}")

        dsplit = np.array_split(df, num_workers)
        def progress(df):
            return [self.process_one(mp_ids, composition, props) for (mp_ids, composition, props) in
                    zip(df['material_id'], tqdm(df['composition']), df[self.prop])]

        cached_data = Parallel(n_jobs=num_workers)(delayed(progress)(split) for split in dsplit)

        for data in cached_data:
            self.cached_data += data
        cnt = 0
        for inx, data in enumerate(self.cached_data):
            if 'graph_issue' in data:
                del self.cached_data[inx]
                cnt += 1
                self.num_train -= 1
                self.num -= 1
        print(f'graph building issue, {cnt} discarded from train data')

    def process_one(self, mp_ids, composition, props):
        formula_arrays = formula_to_spread_graph(composition)
        if formula_arrays is None:
            result_dict = {'graph_issue': True}
            return result_dict
        properties = {'prop': np.array(props)}
        result_dict = {
            'mp_ids': mp_ids,
            'formula_arrays': formula_arrays
        }
        result_dict.update(properties)
        return result_dict

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        atom_type, node_feat, edge_index, elem_weight = data_dict['formula_arrays']
        prop = torch.Tensor(data_dict['prop'])
        data = Data(
            node_features_s=torch.Tensor(node_feat),
            atom_types_s=torch.LongTensor(atom_type),
            edge_index_s=torch.LongTensor(edge_index),
            atom_weights_s=torch.Tensor(elem_weight).reshape(-1, 1),
            y = prop.view(-1,1)
        )
        return data

    def __repr__(self):
        return f"EvalCrystalDataset(len: {len(self.cached_data)})"


def EvalLoader():
    data = EvalDataset(cfg.EVAL.dataset, cfg.PROP)
    data_list = [d for d in data]
    train_set = data_list[:data.num_train]
    valid_set = data_list[data.num_train:data.num_train + data.num_valid]
    test_set = data_list[data.num_train + data.num_valid:]
    train_loader = DataLoader(train_set,
                              batch_size=cfg.EVAL.batch_size,
                              shuffle=True,
                              follow_batch= "atom_types_s"
                              )
    valid_loader = DataLoader(valid_set,
                              batch_size=cfg.EVAL.batch_size,
                              follow_batch= "atom_types_s"
                              )
    test_loader = DataLoader(test_set,
                             batch_size=cfg.EVAL.batch_size,
                             follow_batch= "atom_types_s"
                             )
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    train_loader, valid_loader, test_loader = MaterialLoader()
    # train_loader, valid_loader, test_loader  = EvalLoader()
