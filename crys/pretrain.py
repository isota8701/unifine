import os, pickle
from datetime import datetime
from time import time
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import r2_score
from task import crysDnoise as crysPretrain
from config import cfg

def save_history(history, filename):
    with open(filename, 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)


class EarlyStopping:
    def __init__(self, patience=30):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.message = ''

    def __call__(self, val_loss):
        if val_loss != val_loss:
            self.early_stop = True
            self.message = 'Early stopping: NaN appear'
        elif self.best_score is None:
            self.best_score = val_loss
        elif self.best_score < val_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.message = 'Early stopping: No progress'
        else:
            self.best_score = val_loss
            self.counter = 0


class preTrainer:
    def __init__(self, train_loader, valid_loader, test_loader, args):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = args.device
        self.model = crysPretrain().to(self.device)
        self.optim = optim.AdamW(params=self.model.parameters(),
                                 lr=cfg.TRAIN.lr)
        self.schedular = optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.TRAIN.max_epoch)
        self.min_valid_loss = 1e10
        self.best_epoch = 0
        self.directory = cfg.checkpoint_dir
        name_date = datetime.now().strftime("%m%d")
        data_name = '_'.join([cfg.dataset.split('_')[0], cfg.dataset.split('_')[-1]])
        self.exp_name = f"{args.exp_name}_pretrain_{name_date}_" + data_name
        self.early_stopping = EarlyStopping(patience=cfg.TRAIN.patience)
        self.history = {'train': [], 'valid': [], 'test': []}
        self.history_file = os.path.join(self.directory, 'history_' + self.exp_name + '.pickle')
        ##########################
        self.criterion = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def train(self):
        print("## Start pre-training")
        start_time = time()
        for epoch in range(cfg.TRAIN.max_epoch):
            train_loss, train_loss_dict = self.eval_model(self.train_loader, 'train')
            valid_loss, valid_loss_dict = self.eval_model(self.valid_loader, 'valid')
            self.history['train'].append(train_loss/len(self.train_loader.dataset))
            self.history['valid'].append(valid_loss/len(self.valid_loader.dataset))
            self.schedular.step()
            if valid_loss < self.min_valid_loss:
                torch.save(self.model.state_dict(), os.path.join(self.directory, self.exp_name))
                self.min_valid_loss = valid_loss
                self.best_epoch = epoch
                save_history(self.history, self.history_file)
            self.early_stopping(valid_loss)
            if self.early_stopping.early_stop:
                print(self.early_stopping.message)
                self.model.load_state_dict(torch.load(os.path.join(self.directory, self.exp_name)))
                test_loss, test_loss_dict = self.eval_model(self.test_loader, 'test')
                self.history['test'].append(test_loss / len(self.test_loader.dataset))
                self.history['test'].append(test_loss_dict)
                save_history(self.history, self.history_file)
                print(f'Test Loss:{test_loss / len(self.test_loader.dataset): .4f}')
                print("Test loss details")
                for k, v in test_loss_dict.items():
                    print(f"{str(k)}:{v / len(self.test_loader.dataset): .4f}", end='  ')
                print('')
                break

            if (epoch + 1) % cfg.TRAIN.snapshot_interval == 0:
                print(f'Epoch {epoch + 1}/{cfg.TRAIN.max_epoch}', end='  ')
                print(f'Train loss :{train_loss/len(self.train_loader.dataset):.4f}', end='  ')
                print(f'Valid loss:{valid_loss/len(self.valid_loader.dataset):.4f}', end='  ')
                print(f'Interval time elapsed:{(time() - start_time) / 3600: .4f}')
                print("Train loss details")
                for k,v in train_loss_dict.items():
                    print(f"{str(k)}:{v/len(self.train_loader.dataset): .4f}", end = '  ')
                print('')
                print('-'*100)
                print("Valid loss details")
                for k, v in valid_loss_dict.items():
                    print(f"{str(k)}:{v / len(self.valid_loader.dataset): .4f}", end='  ')
                print('')
                print('-' * 100)

        print("Training done")
        print(f"Best epoch :{self.best_epoch + 1}", end='  ')
        print(f'Best Valid loss:{self.min_valid_loss/len(self.valid_loader.dataset): .4f}')
        end_time = time()
        if epoch + 1 == cfg.TRAIN.max_epoch:
            self.model.load_state_dict(torch.load(os.path.join(self.directory, self.exp_name)))
            test_loss, test_loss_dict = self.eval_model(self.test_loader, 'test')
            self.history['test'].append(test_loss/len(self.test_loader.dataset))
            self.history['test'].append(test_loss_dict)
            save_history(self.history, self.history_file)
            print(f'Test Loss:{test_loss/len(self.test_loader.dataset): .4f}')
            print("Test loss details")
            for k, v in test_loss_dict.items():
                print(f"{str(k)}:{v / len(self.test_loader.dataset): .4f}", end='  ')
            print('')

        print(f"Total time elapsed:{(end_time - start_time) / 3600: .4f}")
        # self.print_result(self.test_loader)

    def eval_model(self, loader, split):
        if split == 'train':
            self.model.train()
            running_loss = 0.
            running_loss_dict = defaultdict(float)
            for gg, fg, gprop, yprop in loader:
                g, lg, fg = gg[0].to(self.device), gg[1].to(self.device), fg.to(self.device)
                gprop, yprop = gprop.to(self.device), yprop.to(self.device)
                loss, loss_dict = self.model((g,lg), fg, gprop, yprop)
                self.optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1000)
                self.optim.step()

                running_loss+=loss.item()*fg.batch_size
                for k, v in loss_dict.items():
                    running_loss_dict[k] +=v.item()*fg.batch_size
            return running_loss, running_loss_dict
        else:
            self.model.eval()
            running_loss = 0.
            running_loss_dict = defaultdict(float)
            for gg, fg, gprop, yprop in loader:
                g, lg, fg = gg[0].to(self.device), gg[1].to(self.device), fg.to(self.device)
                gprop, yprop = gprop.to(self.device), yprop.to(self.device)
                with torch.no_grad():
                    loss, loss_dict = self.model((g,lg), fg, gprop, yprop)
                    running_loss+=loss.item()*fg.batch_size
                    for k, v in loss_dict.items():
                        running_loss_dict[k]+=v.item()*fg.batch_size
            return running_loss, running_loss_dict

    # def print_result(self, loader):
    #     self.model.eval()
    #     outputs = []
    #     labels =  []
    #     running_mae = 0.
    #     running_mse = 0.
    #     for gg,fg,prop in loader:
    #         g, lg = gg[0].to(self.device), gg[1].to(self.device)
    #         fg = fg.to(self.device)
    #         prop = prop.to(self.device)
    #         with torch.no_grad():
    #             ######################TODO################
    #             z1 = self.model.source_encoder(fg)
    #             z2 = self.model.target_encoder((g,lg))
    #             pred = self.model.prop_decoder(z1)
    #             mae = self.mae_loss(pred, prop)
    #             mse = self.criterion(pred, prop)
    #             running_mae+=mae.item()*fg.batch_size
    #             running_mse+=mse.item()*fg.batch_size
    #             outputs.append(pred)
    #             labels.append(prop)
    #     MAE = running_mae / len(loader.dataset)
    #     MSE = running_mse / len(loader.dataset)
    #     RMSE = MSE**0.5
    #     outputs = torch.cat(outputs)
    #     labels = torch.cat(labels)
    #     r2 = self.r2_loss(outputs, labels)
    #     print("Model Performance Metrics:")
    #     print(f"R2 Score: {r2:.4f} ")
    #     print(f"MAE: {MAE:.4f}")
    #     print(f"RMSE: {RMSE:.4f}")
    #
    # def r2_loss(self, out, target):
    #     target_mean = torch.mean(target)
    #     ss_tot = torch.sum((target - target_mean).pow(2))
    #     ss_res = torch.sum((target - out).pow(2))
    #     r2 = 1 - ss_res / ss_tot
    #     return r2.item()



