
import argparse
import os
from data import MaterialsDataloader, EvalLoader
from config import cfg
# from pretrain_em import preTrainer
# from evaluate_em import Evaluator
from pretrain import preTrainer
from evaluate import Evaluator
import random
import torch
import numpy as np

def seed_everything(seed: int = cfg.random_seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def main(args):
    seed_everything()
    # args.pretrain = True
    # args.device = 'cuda:3'
    if args.pretrain:
        train_loader, valid_loader, test_loader = MaterialsDataloader(dataset=cfg.dataset)
        ptrainer = preTrainer(train_loader, valid_loader, test_loader, args)
        ptrainer.train()
        model_path = os.path.join(ptrainer.directory, ptrainer.exp_name)
    else:
        model_path = os.path.join(cfg.checkpoint_dir, cfg.model_path)

    train_loader, valid_loader, test_loader = EvalLoader(dataset = cfg.evalset)
    evaluator = Evaluator(train_loader, valid_loader, test_loader, model_path, args)
    evaluator.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' prediction training')
    parser.add_argument('--pretrain',  action= 'store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0', help="cuda device")
    parser.add_argument('--exp-name', type=str, default='test', help="")
    args = parser.parse_args()
    main(args)


