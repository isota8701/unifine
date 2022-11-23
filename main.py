import argparse
import random
import torch
import numpy as np
import os
from trainer import ModelTrainer
from data import MaterialsDataloader


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def main(args):
    seed_everything(123)
    print(args)
    args.device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    # Prepare dataset
    train_loader, valid_loader, test_loader, test_indices = MaterialsDataloader(args)
    directory = 'checkpoints/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    trainer = ModelTrainer(args, directory, train_loader, valid_loader, test_loader, test_indices)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' prediction training')
    """Experiment setting."""
    parser.add_argument('--exp-name', type=str, default='test', help="")
    parser.add_argument('--num-workers', type=int, default=12, help="")
    parser.add_argument('--dataset', type=str, default='mp_3d_2020', help="data-chemical")
    parser.add_argument('--target', type=str, default='lattice_mat', help="")
    parser.add_argument('--epochs', type=int, default=50, help="")
    parser.add_argument('--num-train', type=int, default=3000, help="")
    parser.add_argument('--num-valid', type=int, default=100, help="")
    parser.add_argument('--num-test', type=int, default=100, help="")
    parser.add_argument('--batch-size', type=int, default=64, help="")
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="")
    parser.add_argument('--weight-decay', type=float, default=0.01, help="")
    parser.add_argument('--max-norm', type=float, default=1000.0, help="")
    parser.add_argument('--scheduler', type=str, default='cosine', help="")
    parser.add_argument('--cutoff', type=float, default=3.0, help="")
    parser.add_argument("--max-neighbors", type = int, default=6)
    parser.add_argument('--device', type=str, default='cuda:0', help="cuda device")
    parser.add_argument('--weights', type = str, default='freeze')
    parser.add_argument('--patience', type = int, default=30)
    '''Model setting'''
    parser.add_argument('--layers', type=int, default=4, help="")
    parser.add_argument('--embedding-type', type=str, default='cgcnn', help="")
    parser.add_argument('--gcn-layers', type=int, default=4, help="")
    parser.add_argument('--atom-input-features', type=int, default=92, help="")
    parser.add_argument('--edge-input-features', type=int, default=80, help="")
    parser.add_argument('--triplet-input-features', type=int, default=40, help="")
    parser.add_argument('--embedding-features', type=int, default=64, help="")
    parser.add_argument('--hidden-features', type=int, default=256, help="")
    parser.add_argument('--output-features', type=int, default=9, help="")
    parser.add_argument('--classification', type=bool, default=False, help="")
    parser.add_argument('--link', type=str, default='identity', help="")
    parser.add_argument('--n-heads', type=int, default=8, help="")
    args = parser.parse_args()
    # Learning
    main(args)
