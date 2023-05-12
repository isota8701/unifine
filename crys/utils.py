
import pickle
import numpy as np
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

import pandas as pd
import random
from jarvis.db.figshare import data as jdata
from config import cfg

class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
            self,
            vmin: float = 0,
            vmax: float = 8,
            bins: int = 40,
            lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )



def refine_materials_project(df, max_atoms):
    df = df[df['atoms'].apply(lambda x: len(x['elements']) < max_atoms)]
    d1 = df[df['full_formula'].duplicated(keep=False) == False]
    d2 = df[df['full_formula'].duplicated(keep=False)]
    min_fe = d2.groupby('full_formula').min('formation_energy_per_atom')['formation_energy_per_atom'].values
    d2 = d2[d2['formation_energy_per_atom'].apply(lambda x: any(x == min_fe))]
    df_ = pd.concat([d1, d2])
    df_ = df_.sample(frac=1).reset_index(drop=True)
    return df_

def refine_oqmd(df ,max_atoms):
    df = df[df['atoms'].apply(lambda x: len(x['elements']) < max_atoms)]
    dd = df.copy()
    def to_formula(alist):
        res = ''
        for e in np.unique(alist):
            res += e.strip()
            res += str(alist.count(e))
        return res

    dd['full_formula'] = df['atoms'].apply(lambda x: to_formula(x['elements']))
    d1 = dd[dd['full_formula'].duplicated(keep=False) == False]
    d2 = dd[dd['full_formula'].duplicated(keep=False)]
    min_fe = d2.groupby('full_formula').min('_oqmd_stability')['_oqmd_stability'].values
    d2 = d2[d2['_oqmd_stability'].apply(lambda x: any(x == min_fe))]
    df_ = pd.concat([d1, d2])
    df_ = df_.sample(frac=1).reset_index(drop=True)
    df_.rename(columns={'_oqmd_stability': 'formation_energy_per_atom', '_oqmd_band_gap': 'band_gap',
                        '_oqmd_delta_e': 'delta_e'}, inplace=True)
    return df_

def refine_dft(df, max_atoms):
    df = df[df['atoms'].apply(lambda x: len(x['elements']) < max_atoms)]
    df = df.rename(columns = {'formula': 'full_formula', 'optb88vdw_bandgap':'band_gap',
                         'formation_energy_peratom':'formation_energy_per_atom'})
    df = df[df[cfg.prop] !='na']
    df['formation_energy_per_atom'] = df['formation_energy_per_atom'].apply(lambda x: float(x))

    d1 = df[df['full_formula'].duplicated(keep=False) == False]
    d2 = df[df['full_formula'].duplicated(keep=False)]
    min_fe = d2.groupby('full_formula').min('formation_energy_per_atom')['formation_energy_per_atom'].values
    d2 = d2[d2['formation_energy_per_atom'].apply(lambda x: any(x == min_fe))]
    df_ = pd.concat([d1, d2])
    df_ = df_.sample(frac=1).reset_index(drop=True)
    return df_

def curate_jdata(dataset_name, max_atoms, cut_data):
    df = pd.DataFrame(jdata(dataset_name))
    if dataset_name.split('_')[0] == 'oqmd':
        df = refine_oqmd(df, max_atoms)
        new_name = 'oqmd'
    elif dataset_name.split('_')[0] == 'mp':
        df = refine_materials_project(df, max_atoms)
        new_name = 'materials-project'
    elif dataset_name == 'dft_2d':
        df = refine_dft(df, max_atoms)
        new_name = 'dft-2d'
    elif dataset_name =='dft_3d':
        df = refine_dft(df, max_atoms)
        new_name = 'dft-3d'
    else:
        assert False, "not available dataset"
    if cut_data:
        print('cutting to small data')
        num = cfg.cut_num
        assert len(df) > num, "less than cut length"
        df = df.iloc[:num]
    print(f"Refined dataset shape {df.shape}")
    # df_joint_test = df.iloc[:10000, :].reset_index(drop = True)
    # df_joint_ptrain = df.iloc[10000:,:].reset_index(drop=True)
    # new_name= f"materials-project_max_atoms_{max_atoms}_dsjoint_len_{len(df_joint_test)}"
    # df_joint_test.to_pickle(cfg.data_dir + f"{new_name}.pkl")
    # new_name= f"materials-project_max_atoms_{max_atoms}_dsjoint_len_{len(df_joint_ptrain)}"
    # df_joint_ptrain.to_pickle(cfg.data_dir + f"{new_name}.pkl")
    new_name+= f"_max_atoms_{max_atoms}_len_{len(df)}"
    df.to_pickle(cfg.data_dir + f"{new_name}.pkl")
    return df

def curate_eval():
    df = pd.read_csv("../data/oqmd-form-enthalpy.csv")
    df.rename(columns = {"composition":"full_formula"}, inplace = True)
    new_name = "oqmd_only-formula_len_256622"
    df.to_pickle(cfg.data_dir + f"{new_name}.pkl")


class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        # X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + 1e-5

    def transform(self, X):
        # X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        # X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )



def get_scaler_from_data_tensor(data_tensor):
    scaler = StandardScalerTorch()
    scaler.fit(data_tensor)
    return scaler

def check_history(filename):
    with open(filename, 'rb') as f:
        history = pickle.load(f)
    print(filename)
    best_inx = np.argmin(history['valid'])
    print(f"Total Epoch:{len(history['train'])}", end = '  ')
    print(f"Best Epoch:{best_inx+1}", end = '  ')
    print(f"Best Train Loss:{history['train'][best_inx]: .4f}", end = '  ')
    print(f"Best Valid Loss:{history['valid'][best_inx]: .4f}")
    print(f"Test Loss:{history['test'][0]: .4f}")
    for k,v in history['test'][1].items():
        print(f"{k} : {v: .4f}")

def plot_history(filename):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    with open(filename, 'rb') as f:
        history = pickle.load(f)
    print(filename)
    plt.figure()
    plt.plot(range(len(history['train'])),history['train'], label = 'train')
    plt.plot(range(len(history['train'])), history['valid'], label = 'valid')
    plt.legend()
    plt.show()





if __name__ == "__main__":
    # cfg.dataset = 'dft_2d'
    # curate_jdata(cfg.dataset, cfg.max_atoms, cfg.cut_data)
    # curate_eval()
    check_history('checkpoints/history_test_pretrain_0511_materials-project_3000.pickle')
    # check_history('checkpoints/history_evaluate_crysVAE_nem_pretrain_0509_materials-project_77153_0511_finetune_formation_10000.pickle')
    # plot_history('checkpoints/history_evaluate_crysVAE_nem_pretrain_0509_materials-project_77153_0511_finetune_formation_10000.pickle')
