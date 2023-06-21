
import argparse
import os
from data import MaterialsDataloader, EvalLoader
from config import cfg
# from pretrain_em import preTrainer
# from evaluate_em import Evaluator
from pretrain import preTrainer, Trainer
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
    if args.pretrain:
        train_loader, valid_loader, test_loader = MaterialsDataloader(args, dataset=cfg.dataset)
        ptrainer = preTrainer(train_loader, valid_loader, test_loader, args)
        ptrainer.train()
        model_path = os.path.join(ptrainer.directory, ptrainer.exp_name)
    else:
        model_path = os.path.join(cfg.checkpoint_dir, cfg.model_name)

    train_loader, valid_loader, test_loader = EvalLoader(args,dataset = cfg.evalset)
    evaluator = Evaluator(train_loader, valid_loader, test_loader, model_path, args)
    evaluator.train()

def kd(args):
    seed_everything()
    train_loader, valid_loader, test_loader = MaterialsDataloader(args,dataset=cfg.dataset)
    trainer = Trainer(train_loader, valid_loader, test_loader, args)
    trainer.train()
    model_path = os.path.join(trainer.directory, trainer.exp_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' prediction training')
    parser.add_argument('--pretrain',  action= 'store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0', help="cuda device")
    parser.add_argument('--exp-name', type=str, default='test', help="")
    parser.add_argument('--target', type = str, default = 'formation_energy_per_atom')
    parser.add_argument('--weight', type = str, default='rand-init')
    parser.add_argument('--fg', type = str, default= "mask", help = 'formula graph style')
    parser.add_argument('--kd', type = str, default='student')
    args = parser.parse_args()
    main(args)
    # kd(args)

