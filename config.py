import numpy as np
from easydict import EasyDict as edict

'''
base download dataset name
"mp_3d_2020"
"oqmd_3d_no_cfid" --> "_oqmd_stability" data preprocession issue

curated dataset name
"mp_3d_2020_maxA20_N48840"
"oqmd_3d_no_cfid_maxA20_N412618"

fast test dataset name
"mp_3d_2020_maxA20_N48840_CUT_3600"
"mp_3d_2020_maxA20_N48840_CUT_1100"
'''
__C = edict()
cfg = __C
# loading from path
__C.DATA_DIR = '../data/'
__C.FROM_PATH = True
__C.CUT_DATA = False
__C.DATASET_NAME = "mp_3d_2020_maxA20_N48840_CUT_1100"
__C.PROP = "formation_energy_per_atom"
__C.NUM_TRAIN = 1000
__C.NUM_VALID = 50
__C.NUM_TEST = 50
__C.TRAIN_RATIO = 0.9
__C.VALID_RATIO = 0.05
__C.TEST_RATIO = 0.05
__C.MAX_ATOMS = 20
__C.MAX_ATOMIC_NUM = 100
__C.CUDA = True
__C.RANDOM_SEED = 122
__C.WORKERS = 12
__C.CHECKPOINT_DIR = '../checkpoints/'
__C.CHECKPOINT_NAME = ''
__C.SCALER_DIR = '../data/'
__C.WEIGHTS = 'finetune'
__C.DEVICE = 'cuda:0'
# Test options
__C.TEST = edict()


# Training options
__C.TRAIN = edict()
__C.TRAIN.max_epoch = 20
__C.TRAIN.batch_size = 64
__C.TRAIN.snapshot_interval =1
__C.TRAIN.lr = 0.001
__C.TRAIN.patience = 40


__C.EVAL = edict()
__C.EVAL.dataset = "EVAL_mp_3d_2020_maxA20_N21156_formation_energy_per_atom.csv"
__C.EVAL.lr_head = 0.001
__C.EVAL.lr_backbone = 0.001
__C.EVAL.max_epoch = 20
__C.EVAL.batch_size = 256
__C.EVAL.snapshot_interval = 1

# Modal options

__C.GNN = edict()
__C.GNN.encoder = ''
__C.GNN.hidden_dim = 256
__C.GNN.atom_input_dim = 92

__C.FORMULA = edict()
__C.FORMULA.layers = 3
__C.FORMULA.atom_input_dim = 92
__C.FORMULA.hidden_dim = 256
__C.FORMULA.edge_dim = 64
__C.FORMULA.output_dim = 1
__C.FORMULA.n_heads = 4
__C.FORMULA.encoder = ''

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

__C.DECODER = edict()
__C.DECODER.latent_dim = 512



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