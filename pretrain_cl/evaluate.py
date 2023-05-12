import pickle
import numpy as np
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_add_pool
import os
from pretrain import crysCL
from config import cfg, cfg_from_file
from time import time
from sklearn.metrics import r2_score

def main():
    # cfg_from_file("default.yaml")
    from pretrain_cl.data import EvalLoader
    train_loader, valid_loader, test_loader = EvalLoader()
    load_pth = os.path.join(cfg.CHECKPOINT_DIR, cfg.model_path)
    evaluator = Evaluator(train_loader, valid_loader, test_loader, load_pth)
    evaluator.train()

def save_history(history, filename):
    with open(filename, 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

class Decoder(nn.Module):
    def __init__(self,
                 load_exp_name = None):
        super().__init__()
        self.directory = cfg.CHECKPOINT_DIR
        self.exp_name = load_exp_name
        model = crysCL()
        if cfg.WEIGHTS not in ['freeze', 'finetune', 'transfer', 'rand_init']:
            assert False, "Unspecified Evaluation Type"
        if cfg.WEIGHTS in ['freeze', 'finetune']:
            model.load_state_dict(torch.load(os.path.join(self.directory, self.exp_name)))
        self.backbone = model.source_encoder
        self.head = nn.Linear(cfg.FORMULA.hidden_dim, 1)
        self.head.weight.data.normal_(mean = 0., std = 0.01)
        self.head.bias.data.zero_()

    def forward(self, data):
        z, mu,logvar = self.backbone(data)
        if "atom_types_s_batch" in data.keys:
            batch = data.atom_types_s_batch
        elif "batch" in data.keys:
            batch = data.batch
        z_G = global_add_pool(z, batch)
        out = self.head(z_G)
        return out

class Evaluator:
    def __init__(self, train_loader , valid_loader, test_loader, load_exp_name):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.directory = cfg.CHECKPOINT_DIR
        name_date = datetime.now().strftime("%m%d")
        self.exp_name = f"evaluate_{name_date}_{cfg.WEIGHTS}_{cfg.PROP}_" + cfg.DATASET_NAME.split('_')[-1]
        self.decoder = Decoder(load_exp_name).to(cfg.DEVICE)
        self.min_valid_loss = 1e10
        self.best_epoch = 0
        if cfg.WEIGHTS =='freeze':
            self.decoder.backbone.requires_grad_(False)
            self.decoder.head.requires_grad_(True)
        self.criterion = nn.MSELoss()
        param_groups = [dict(params = self.decoder.head.parameters(), lr = cfg.EVAL.lr_head)]
        if cfg.WEIGHTS in ['finetune', 'rand_init']:
            param_groups.append(dict(params = self.decoder.backbone.parameters(), lr = cfg.EVAL.lr_backbone))
        self.optimizer = torch.optim.AdamW(param_groups)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg.EVAL.max_epoch)
        self.history = {'data': [], 'valid': [], 'test': []}
        self.history_file = os.path.join(self.directory, 'history_' + self.exp_name + '.pickle')
        self.mae_loss = nn.L1Loss()

    def train(self):
        print(f"## Start linear evaluation (weights:{cfg.WEIGHTS}, target property:{cfg.PROP})")
        start_time = time()
        for epoch in range(cfg.EVAL.max_epoch):
            if cfg.WEIGHTS == 'freeze':
                self.decoder.eval()
            else:
                self.decoder.train()
            train_loss = self.eval_model(self.train_loader, 'data')
            valid_loss = self.eval_model(self.valid_loader, 'valid')
            self.history['data'].append(train_loss)
            self.history['valid'].append(valid_loss)
            self.scheduler.step()
            if valid_loss < self.min_valid_loss:
                torch.save(self.decoder.state_dict(), os.path.join(self.directory, self.exp_name))
                self.min_valid_loss = valid_loss
                self.best_epoch = epoch
                save_history(self.history, self.history_file)
            if (epoch+1) % cfg.EVAL.snapshot_interval ==0:
                print(f'Epoch {epoch + 1}/{cfg.EVAL.max_epoch}', end='  ')
                print(f'Train loss :{train_loss:.4f}', end='  ')
                print(f'Valid loss:{valid_loss:.4f}', end='  ')
                print(f'Interval time elapsed:{(time() - start_time) / 3600: .4f}')
        print("Training done")
        print(f"Best epoch :{self.best_epoch + 1}", end='  ')
        print(f'Best Valid loss:{self.min_valid_loss: .4f}')
        end_time = time()
        if epoch + 1 == cfg.EVAL.max_epoch:
            self.decoder.load_state_dict(torch.load(os.path.join(self.directory, self.exp_name)))
            test_loss = self.eval_model(self.test_loader, 'test')
            self.history['test'].append(test_loss)
            save_history(self.history, self.history_file)
            print(f'Test Loss:{test_loss: .4f}')
            self.print_result(self.test_loader)

    def eval_model(self, loader, split):
        if split == 'data':
            running_loss = []
            for batch in loader:
                batch = batch.to(cfg.DEVICE)
                out = self.decoder(batch)
                loss = self.criterion(out, batch.y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss.append(loss.item())
            return np.mean(running_loss)
        else:
            self.decoder.eval()
            running_loss = []
            for batch in loader:
                batch = batch.to(cfg.DEVICE)
                with torch.no_grad():
                    out = self.decoder(batch)
                    loss = self.criterion(out, batch.y)
                    running_loss.append(loss.item())
            return np.mean(running_loss)

    def print_result(self, loader):
        self.decoder.eval()
        running_loss1, running_loss2 = [], []
        for batch in loader:
            batch = batch.to(cfg.DEVICE)
            with torch.no_grad():
                pred = self.decoder(batch)
                mae = self.mae_loss(pred, batch.y)
                mse = self.criterion(pred, batch.y)
                running_loss1.append(mae.item())
                running_loss2.append(mse.item())
        Mae = np.mean(running_loss1)
        Mse = np.mean(running_loss2)
        Rmse = np.sqrt(Mse)
        r2 = r2_score(batch.y.detach().cpu().numpy(), pred.detach().cpu().numpy())
        print("Model Performance Metrics:")
        print(f"R2 Score: {r2:.4f} ")
        print(f"MAE: {Mae:.4f}")
        print(f"RMSE: {Rmse:.4f}")







if __name__ == "__main__":
    main()