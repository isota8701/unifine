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
from cdvae import target_solver
from gnn_2d import Encoder as source_encoder
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

        self.source_encoder = source_encoder().to(self.device) # z
        self.proj = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)).to(self.device)
        self.invp = nn.Sequential(nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim),
                                  nn.BatchNorm1d(cfg.GNN.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim)).to(self.device)

        self.target_solver = target_solver().to(self.device)# output z & 3d property loss (y, lattice...)
        self.source_optim = optim.AdamW(params=list(self.source_encoder.parameters()) +list(self.proj.parameters()) +list(self.invp.parameters()),
                                        lr = cfg.TRAIN.lr)
        self.target_optim = optim.AdamW(params= self.target_solver.parameters(),
                                        lr = cfg.TRAIN.lr)

        self.source_schedular = optim.lr_scheduler.CosineAnnealingLR(self.source_optim, cfg.TRAIN.max_epoch)
        self.target_schedular = optim.lr_scheduler.CosineAnnealingLR(self.target_optim, cfg.TRAIN.max_epoch)
        self.min_valid_loss = 1e10
        self.best_epoch = 0
        self.directory = cfg.checkpoint_dir
        name_date = datetime.now().strftime("%m%d")
        data_name = '_'.join([cfg.dataset.split('_')[0], cfg.dataset.split('_')[-1]])
        self.exp_name = f"{args.exp_name}_source_pretrain_{name_date}_" + data_name
        self.target_exp_name = f"{args.exp_name}_target_pretrain_{name_date}_" + data_name
        self.early_stopping = EarlyStopping(patience=cfg.TRAIN.patience)
        self.history = {'train': [], 'valid': [], 'test': []}
        self.history_file = os.path.join(self.directory, 'history_' + self.exp_name + '.pickle')
        # cos loss
        self.cos = nn.CosineSimilarity(dim = 1)
        self.mae_loss = nn.L1Loss()
        # cycle loss
        self.mse = nn.MSELoss()



    def train(self):
        print("## Start pre-training")
        start_time = time()
        for epoch in range(cfg.TRAIN.max_epoch):
            train_loss, train_loss_dict = self.eval_model(self.train_loader, 'train')
            valid_loss, valid_loss_dict = self.eval_model(self.valid_loader, 'valid')
            self.history['train'].append(train_loss/len(self.train_loader.dataset))
            self.history['valid'].append(valid_loss/len(self.valid_loader.dataset))
            self.target_schedular.step()
            self.source_schedular.step()
            if valid_loss < self.min_valid_loss:
                torch.save({'source_enc': self.source_encoder.state_dict(),
                            'proj': self.proj.state_dict(),
                            'invp':self.invp.state_dict()}, os.path.join(self.directory, self.exp_name))
                torch.save(self.target_solver.state_dict(), os.path.join(self.directory, self.target_exp_name))
                self.min_valid_loss = valid_loss
                self.best_epoch = epoch
                save_history(self.history, self.history_file)
            self.early_stopping(valid_loss)
            if self.early_stopping.early_stop:
                print(self.early_stopping.message)
                checkpoint = torch.load(os.path.join(self.directory, self.exp_name))
                self.source_encoder.load_state_dict(checkpoint['source_enc'])
                self.proj.load_state_dict(checkpoint['proj'])
                self.invp.load_state_dict(checkpoint['invp'])
                self.target_solver.load_state_dict(torch.load(os.path.join(self.directory, self.target_exp_name)))
                test_loss, test_loss_dict = self.eval_model(self.test_loader, 'test')
                self.history['test'].append(test_loss/len(self.test_loader.dataset))
                self.history['test'].append(test_loss_dict)
                save_history(self.history, self.history_file)
                print(f'Test Loss:{test_loss/len(self.test_loader.dataset): .4f}')
                print("Test loss details")
                for k,v in test_loss_dict.items():
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
                for k,v in valid_loss_dict.items():
                    print(f"{str(k)}:{v/len(self.valid_loader.dataset): .4f}", end = '  ')
                print('')
                print('-'*100)

        print("Training done")
        print(f"Best epoch :{self.best_epoch + 1}", end='  ')
        print(f'Best Valid loss:{self.min_valid_loss/len(self.valid_loader.dataset): .4f}')
        end_time = time()
        if epoch + 1 == cfg.TRAIN.max_epoch:
            checkpoint = torch.load(os.path.join(self.directory, self.exp_name))
            self.source_encoder.load_state_dict(checkpoint['source_enc'])
            self.proj.load_state_dict(checkpoint['proj'])
            self.invp.load_state_dict(checkpoint['invp'])
            self.target_solver.load_state_dict(torch.load(os.path.join(self.directory, self.target_exp_name)))
            test_loss, test_loss_dict = self.eval_model(self.test_loader, 'test')
            self.history['test'].append(test_loss/len(self.test_loader.dataset))
            self.history['test'].append(test_loss_dict)
            save_history(self.history, self.history_file)
            print(f'Test Loss:{test_loss/len(self.test_loader.dataset): .4f}')
            print("Test loss details")
            for k,v in test_loss_dict.items():
                print(f"{str(k)}:{v / len(self.test_loader.dataset): .4f}", end='  ')
            print('')
        print(f"Total time elapsed:{(end_time - start_time) / 3600: .4f}")
        # self.print_result(self.test_loader)

    def eval_model(self, loader, split):
        if split == 'train':
            self.source_encoder.train()
            self.proj.train()
            self.invp.train()
            self.target_solver.train()
            running_loss = 0.
            running_loss_dict = defaultdict(float)
            for gg, fg, gprop, yprop in loader:
                g, lg, fg = gg[0].to(self.device), gg[1].to(self.device), fg.to(self.device)
                gprop, yprop = gprop.to(self.device), yprop.to(self.device)
                zt, target_loss, loss_dict = self.target_solver((g, lg), gprop,yprop)
                self.target_optim.zero_grad()
                target_loss.backward()
                torch.nn.utils.clip_grad_norm(self.target_solver.parameters(), max_norm=1000)
                self.target_optim.step()
                self.source_optim.zero_grad()

                zs = self.source_encoder(fg)
                # cosine similarity
                # cos_loss = -(self.cos(zt.detach(), zs).mean())
                # cos_loss.backward()

                # cycle loss
                loss_p = self.mse(self.proj(zs), zt.detach())
                loss_i = self.mse(zs, self.invp(zt.detach()))
                loss_c = self.mse(self.invp(self.proj(zs)), zs)
                cycle_loss = loss_p + loss_i + loss_c
                cycle_loss.backward()
                torch.nn.utils.clip_grad_norm(self.source_encoder.parameters(), max_norm=1000)
                self.source_optim.step()
                loss_dict.update({
                    # "cosine_emb_loss": cos_loss,
                    'cycle_loss':cycle_loss,
                })
                loss = target_loss + cycle_loss

                # info NCE
                # cl_loss = self.target_solver.info_nce_loss(zs, zt.detach())
                # cl_loss.backward()
                # self.source_optim.step()
                # loss_dict.update({'contrastive_loss': cl_loss})
                # loss = target_loss + cl_loss

                running_loss+=loss.item()*fg.batch_size
                for k, v in loss_dict.items():
                    running_loss_dict[k] +=v.item()*fg.batch_size
            return running_loss, running_loss_dict
        else:
            self.source_encoder.eval()
            self.proj.eval()
            self.invp.eval()
            self.target_solver.eval()
            running_loss = 0.
            running_loss_dict = defaultdict(float)
            for gg, fg, gprop, yprop in loader:
                g, lg, fg = gg[0].to(self.device), gg[1].to(self.device), fg.to(self.device)
                gprop, yprop = gprop.to(self.device), yprop.to(self.device)
                with torch.no_grad():
                    zt, target_loss, loss_dict = self.target_solver((g, lg), gprop, yprop)
                    zs = self.source_encoder(fg)

                    # cosine similiarity
                    # cos_loss = -(self.cos(zt.detach(), zs).mean())
                    # loss_dict.update({"cosine_emb_loss": cos_loss})
                    # loss = target_loss + cos_loss

                    # cycle loss
                    loss_p = self.mse(self.proj(zs), zt.detach())
                    loss_i = self.mse(zs, self.invp(zt.detach()))
                    loss_c = self.mse(self.invp(self.proj(zs)), zs)
                    cycle_loss = loss_p + loss_i + loss_c
                    loss_dict.update({
                        # "cosine_emb_loss": cos_loss,
                        'cycle_loss': cycle_loss,
                    })
                    loss = target_loss + cycle_loss
                    # Info NCE
                    # cl_loss = self.target_solver.info_nce_loss(zs, zt.detach())
                    # loss_dict.update({'contrastive_loss': cl_loss})
                    # loss = target_loss + cl_loss

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



