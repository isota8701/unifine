import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: birds
__C.DATASET_NAME = 'mp_3d_2020'
__C.DATA_DIR = './data/'
__C.NUM_TRAIN = 1000
__C.NUM_VALID = 50
__C.NUM_TEST = 50
__C.MAX_ATOMS = 20
__C.MAX_ATOMIC_NUM = 100
__C.DEVICE = 'cuda:0'
__C.CUDA = True
__C.RANDOM_SEED = 321
__C.WORKERS = 12
__C.CHECKPOINT_DIR = './checkpoints/'
__C.CHECKPOINT_NAME = ''
__C.SCALER_DIR = './data/'

# Test options
__C.TEST = edict()


# Training options
__C.TRAIN = edict()
__C.TRAIN.max_epoch = 100
__C.TRAIN.batch_size = 25
__C.TRAIN.snapshot_interval = 10



# Modal options

__C.GNN = edict()
__C.GNN.encoder = ''

__C.FORMULA = edict()
__C.FORMULA.layers = 4
__C.FORMULA.atom_input_dim = 92
__C.FORMULA.hidden_dim = 256
__C.FORMULA.output_dim = 1
__C.FORMULA.n_heads = 4
__C.FORMULA.encoder = ''

# Modal options
__C.VAE = edict()
__C.VAE.cost_natom = 1.
__C.VAE.cost_coord = 10.
__C.VAE.cost_type = 1.
__C.VAE.cost_lattice = 10.
__C.VAE.cost_composition = 1.
__C.VAE.cost_property =  1.
__C.VAE.cost_kld = 0.01
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

__C.OPTIM = edict()
__C.OPTIM.lr = 0.001


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