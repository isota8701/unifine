import numpy as np
from easydict import EasyDict as edict
'''
jdata name
materials project: mp_3d_2020
oqmd: oqmd_3d_no_cfid
dft: dft_3d  dft_2d
'''
__C = edict()
cfg = __C
# loading from path
__C.data_dir = './data/'
__C.cut_data = False
# cuda0
# __C.dataset = "materials-project_max_atoms_50_dsjoint_len_72902"
__C.dataset = "materials-project_max_atoms_20_len_3000"
__C.evalset = "materials-project_max_atoms_50_dsjoint_len_10000"
__C.prop = "formation_energy_per_atom"  # 'e_above_hull'
__C.model_name = "no_name"
__C.weights = "finetune"
__C.cut_num = 10000
__C.train_ratio = 0.9
__C.valid_ratio = 0.05
__C.test_ratio = 0.05
__C.max_atoms = 50
__C.max_atomic_num = 100
__C.cuda = True
__C.random_seed = 1
__C.workers = 3
__C.checkpoint_dir = './checkpoints/'
__C.cutoff = 3.
__C.max_neighbors = 6
__C.max_norm = 1000.
__C.temperature = 0.07


# Test options
__C.TEST = edict()

# Training options
__C.TRAIN = edict()
__C.TRAIN.max_epoch = 2
__C.TRAIN.batch_size = 32
__C.TRAIN.snapshot_interval =10
__C.TRAIN.lr = 0.001
__C.TRAIN.patience = 50


__C.EVAL = edict()
__C.EVAL.dataset = ""
__C.EVAL.lr_head = 0.01
__C.EVAL.lr_backbone = 0.001
__C.EVAL.max_epoch = 1
__C.EVAL.batch_size = 128
__C.EVAL.snapshot_interval = 10

# Modal options

__C.GNN = edict()
__C.GNN.encoder = ''
__C.GNN.hidden_dim = 256
__C.GNN.atom_input_dim = 92
__C.GNN.edge_input_dim = 80
__C.GNN.triplet_input_dim = 40
__C.GNN.embedding_dim = 64
__C.GNN.alignn_layers = 4
__C.GNN.gcn_layers = 4
__C.GNN.roost_layers = 6
__C.GNN.num_heads = 8
__C.GNN.output_dim = 1

# Modal options
__C.VAE = edict()
__C.VAE.cost_natom = 1.
__C.VAE.cost_coord = 1.
__C.VAE.cost_type = 1.
__C.VAE.cost_lattice = 1.
__C.VAE.cost_composition = 1.
__C.VAE.cost_property =  1.
__C.VAE.cost_kld = 1.
__C.VAE.sigma_begin = 10.
__C.VAE.sigma_end = 0.01
__C.VAE.type_sigma_begin = 5.
__C.VAE.type_sigma_end = 0.01
__C.VAE.num_noise_level = 50
__C.VAE.predict_property = False
__C.VAE.max_neighbors = 20
__C.VAE.radius = 7
__C.VAE.hidden_dim = 256




def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering theoptions in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    import yaml
    with open(filename, 'r') as f:
        # yaml_cfg = edict(yaml.load(f))
        yaml_cfg = edict(yaml.safe_load(f))
        # IF ERROR, CHANGE ABOVE LINE TO "yaml_cfg = edict(yaml.safe_load(f))"
    _merge_a_into_b(yaml_cfg, __C)