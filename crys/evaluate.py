import pickle
import numpy as np
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
from task import crysVQVAE as crysPretrain
from config import cfg
from time import time
from dgl.nn.pytorch.glob import AvgPooling

def save_history(history, filename):
    with open(filename, 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)


class Evaluator:
    def __init__(self, train_loader , valid_loader, test_loader, load_model_path, args):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = args.device
        trained_model = crysPretrain()
        if cfg.weights not in ['freeze', 'finetune', 'supervised', 'rand-init', '3d']:
            assert False, "Unspecified Evaluation Type"
        if cfg.weights in ['freeze', 'finetune', 'supervised']:
            trained_model.load_state_dict(torch.load(load_model_path))
        self.pooling = AvgPooling()
        # base
        # self.backbone = nn.Sequential(trained_model.source_encoder,trained_model.proj)

        # vq
        self.enc = trained_model.source_encoder
        self.vq = trained_model.vq_layer
        self.backbone = nn.ModuleList([self.enc,self.vq])


        if cfg.weights == '3d':
            self.backbone = trained_model.target_encoder
        elif cfg.weights == 'supervised':
            self.backbone = trained_model.source_encoder

        ###################################
        # base
        self.head = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.output_dim)
        self.head.weight.data.normal_(mean=0., std=0.01)
        self.head.bias.data.zero_()
        if (cfg.weights =='freeze'):
            self.backbone.requires_grad_(False)
            self.head.requires_grad_(True)


        param_groups = [dict(params = self.head.parameters(), lr = cfg.EVAL.lr_head)]
        if cfg.weights in ['finetune', 'rand-init', 'supervised', '3d']:
            param_groups.append(dict(params = self.backbone.parameters(), lr = cfg.EVAL.lr_backbone))
        self.optimizer = torch.optim.AdamW(param_groups)

        # base
        # self.model = nn.Sequential(self.backbone, self.head)
        # vqvae
        self.model = self.backbone.append(self.head)

        self.model = self.model.to(self.device)
        self.min_valid_loss = 1e10
        self.best_epoch = 0
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg.EVAL.max_epoch)

        self.directory = cfg.checkpoint_dir
        name_date = datetime.now().strftime("%m%d")
        if args.pretrain:
            self.exp_name = f"evaluate_{args.exp_name}_{name_date}_{cfg.weights}_{cfg.prop.split('_')[0]}_" + cfg.evalset.split('_')[-1]
        else:
            call_name = cfg.model_name
            call_name = call_name.split("pretrain")[0]
            self.exp_name = f"evaluate_{call_name}_{name_date}_{cfg.weights}_{cfg.prop.split('_')[0]}_" + cfg.evalset.split('_')[-1]

        self.history = {'train': [], 'valid': [], 'test': []}
        self.history_file = os.path.join(self.directory, 'history_' + self.exp_name + '.pickle')
        self.mae_loss = nn.L1Loss()

    def train(self):
        print(f"## Start linear evaluation (weights:{cfg.weights}, target property:{cfg.prop})")
        start_time = time()
        for epoch in range(cfg.EVAL.max_epoch):
            if cfg.weights == 'freeze':
                self.model.eval()
            else:
                self.model.train()
            train_loss = self.eval_model(self.train_loader, 'train')
            valid_loss = self.eval_model(self.valid_loader, 'valid')
            self.history['train'].append(train_loss/len(self.train_loader.dataset))
            self.history['valid'].append(valid_loss/len(self.valid_loader.dataset))
            self.scheduler.step()
            if valid_loss < self.min_valid_loss:
                torch.save(self.model.state_dict(), os.path.join(self.directory, self.exp_name))
                self.min_valid_loss = valid_loss
                self.best_epoch = epoch
                save_history(self.history, self.history_file)
            if (epoch+1) % cfg.EVAL.snapshot_interval ==0:
                print(f'Epoch {epoch + 1}/{cfg.EVAL.max_epoch}', end='  ')
                print(f'Train loss :{train_loss/len(self.train_loader.dataset):.4f}', end='  ')
                print(f'Valid loss:{valid_loss/len(self.valid_loader.dataset):.4f}', end='  ')
                print(f'Interval time elapsed:{(time() - start_time) / 3600: .4f}')
        print("Training done")
        print(f"Best epoch :{self.best_epoch + 1}", end='  ')
        print(f'Best Valid loss:{self.min_valid_loss/len(self.valid_loader.dataset): .4f}')
        end_time = time()
        if epoch + 1 == cfg.EVAL.max_epoch:
            self.model.load_state_dict(torch.load(os.path.join(self.directory, self.exp_name)))
            test_loss = self.eval_model(self.test_loader, 'test')
            self.history['test'].append(test_loss/len(self.test_loader.dataset))
            save_history(self.history, self.history_file)
            print(f'Test Loss:{test_loss/len(self.test_loader.dataset): .4f}')

        print(f"Total time elapsed:{(end_time - start_time) / 3600: .4f}")
        r2, mae, rmse = self.print_result(self.test_loader)
        self.history['test'].append({'r2':r2, 'mae':mae, 'rmse':rmse})
        save_history(self.history, self.history_file)

    def eval_model(self, loader, split):
        if split == 'train':
            running_loss =  0.
            for batch in loader:
                if cfg.weights == '3d':
                    gg, fg, prop = batch
                    g, lg =gg[0].to(self.device), gg[1].to(self.device)
                    input_g = (g, lg)
                else:
                    fg, prop = batch
                    fg = fg.to(self.device)
                    input_g = fg
                prop = prop.to(self.device)
                # base
                # out = self.model(input_g)

                # vqvae
                zs = self.enc(input_g)
                q, vq_loss = self.vq.eval_forward(zs)
                q = self.pooling(input_g, q)
                out = self.head(q)
                loss = self.criterion(out, prop)
                loss+= vq_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss+=loss.item()*fg.batch_size
            return running_loss
        else:
            self.model.eval()
            running_loss = 0.
            for batch in loader:
                if cfg.weights == '3d':
                    gg, fg, prop = batch
                    g, lg = gg[0].to(self.device), gg[1].to(self.device)
                    input_g = (g, lg)
                else:
                    fg, prop = batch
                    fg = fg.to(self.device)
                    input_g = fg
                prop = prop.to(self.device)
                with torch.no_grad():
                    # out = self.model(input_g)
                    # vqvae
                    zs = self.enc(input_g)

                    q, vq_loss = self.vq.eval_forward(zs)

                    q = self.pooling(input_g, q)
                    out = self.head(q)

                    loss = self.criterion(out, prop)
                    running_loss+=loss.item()*fg.batch_size
            return running_loss

    def print_result(self, loader):
        self.model.eval()
        outputs = []
        labels =  []
        running_mae = 0.
        running_mse = 0.
        for batch in loader:
            if cfg.weights == '3d':
                gg, fg, prop = batch
                g, lg = gg[0].to(self.device), gg[1].to(self.device)
                input_g = (g, lg)

            else:
                fg, prop = batch
                fg = fg.to(self.device)
                input_g = fg
            prop = prop.to(self.device)
            with torch.no_grad():
                # out = self.model(input_g)
                # vqvae
                zs = self.enc(input_g)

                q, vq_loss = self.vq.eval_forward(zs)

                q = self.pooling(input_g,q)
                out = self.head(q)

                mae = self.mae_loss(out, prop)
                mse = self.criterion(out, prop)
                running_mae+=mae.item()*fg.batch_size
                running_mse+=mse.item()*fg.batch_size
                outputs.append(out)
                labels.append(prop)
        MAE = running_mae / len(loader.dataset)
        MSE = running_mse / len(loader.dataset)
        RMSE = MSE**0.5
        outputs = torch.cat(outputs)
        labels = torch.cat(labels)
        r2 = self.r2_loss(outputs, labels)
        print("Model Performance Metrics:")
        print(f"R2 Score: {r2:.4f} ")
        print(f"MAE: {MAE:.4f}")
        print(f"RMSE: {RMSE:.4f}")
        return r2, MAE, RMSE

    def r2_loss(self, out, target):
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean).pow(2))
        ss_res = torch.sum((target - out).pow(2))
        r2 = 1 - ss_res / ss_tot
        return r2.item()

