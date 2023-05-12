import os
from collections import defaultdict
from datetime import datetime
import numpy as np
import pickle
from tqdm import tqdm
from time import time
from config import cfg, cfg_from_file
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch_geometric.nn import global_add_pool
from torch import Tensor
from torch_geometric.utils import to_dense_adj, unbatch, unbatch_edge_index
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import NoneType  # noqa
from dimenet import DimeNetPlusPlusWrap as DimeNet
from formula import FormulaNet
from denoise import Denoiser
from pytorch_metric_learning.losses import NTXentLoss

from sklearn.metrics import r2_score


def main():
    # cfg_from_file("default.yaml")
    from pretrain_cl.data import MaterialLoader
    train_loader, valid_loader, test_loader = MaterialLoader()
    ptrainer = preTrainer(train_loader, valid_loader, test_loader)
    ptrainer.train()


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


class crysAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DimeNet()
        self.fc_atom_feature = nn.Linear(cfg.GNN.hidden_dim, cfg.GNN.atom_input_dim)
        self.bi_adj = nn.Bilinear(cfg.GNN.hidden_dim, cfg.GNN.hidden_dim, cfg.MAX_ATOMS)
        self.fc1 = nn.Linear(cfg.MAX_ATOMS, cfg.MAX_ATOMS)
        self.fc_sg = nn.Linear(cfg.GNN.hidden_dim, 230)

    def forward(self, data):
        edge_index = data.edge_index_t
        batch = data.atom_types_t_batch
        z = self.encoder(data)

        # z = F.normalize(z, dim=1)
        z_G = global_add_pool(z, batch)
        # z_G = F.normalize(z_G, dim = -1)
        atom_feat = self.fc_atom_feature(z)
        sg_pred = F.log_softmax(self.fc_sg(z_G), dim=1)

        ee = unbatch_edge_index(edge_index, batch)
        zz = unbatch(z, batch)
        edge_probs, adj_list = [], []
        for uz, ue in zip(zz, ee):
            N, dim = uz.shape[0], uz.shape[1]
            z_repeat = uz.repeat(N, 1, 1)
            z_repeat = z_repeat.contiguous().view(-1, dim)

            z_expand = torch.unsqueeze(uz, 1).expand(N, N, dim)
            z_expand = z_expand.contiguous().view(-1, dim)

            edge_p = self.bi_adj(z_expand, z_repeat)
            edge_p = self.fc1(edge_p)
            edge_p = F.log_softmax(edge_p, dim=1)
            edge_probs.append(edge_p)

            adj = to_dense_adj(ue).flatten()
            adj_list.append(adj)

        edge_probs = torch.cat(edge_probs, dim=0).to(cfg.DEVICE)
        adjs = torch.cat(adj_list).type(torch.LongTensor).to(cfg.DEVICE)
        return (edge_probs, adjs), atom_feat, sg_pred, z_G, z


class crysCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = FormulaNet().to(cfg.DEVICE)
        self.target_encoder = crysAE().to(cfg.DEVICE)
        self.ntx = NTXentLoss()
        self.mse = nn.MSELoss()
        self.fc_out = nn.Linear(cfg.FORMULA.hidden_dim, 1).to(cfg.DEVICE)

        self.test_cls = nn.Parameter(torch.randn(1,64))

    def forward(self, data):
        z1, mu, logvar = self.source_encoder(data)
        # test_token = repeat(self.test_cls, '() e' -> 'n e', n = z1.shape[0])
        (edge_probs, adjs), atom_feat_p, sg_p, z_G, z2 = self.target_encoder(data)

        loss_atom_feat_recon = F.binary_cross_entropy_with_logits(atom_feat_p,
                                                                  data.node_features_t)
        # y_loss = self.mse(self.fc_out(z_G), data.y_t)
        z1_G = global_add_pool(z1, data.atom_types_s_batch)
        z1_G = F.normalize(z1_G, dim=-1)
        pos_weight = torch.ones(cfg.MAX_ATOMS).to(cfg.DEVICE)
        pos_weight[0] = 0.1

        loss_adj_recon = F.nll_loss(edge_probs, adjs, weight=pos_weight)
        loss_sg_recon = F.nll_loss(sg_p, data.spacegroup_no_t)
        sg_nce_loss = self.ntx(z_G, data.spacegroup_t)
        sg_nce_loss += self.ntx(z1_G, data.spacegroup_t)
        Cl_loss, Cl_acc = dual_CL(z1_G, z_G)
        emb_loss = self.mse(z1, z2)
        kld_loss = self.kld_loss(mu, logvar)

        loss = loss_atom_feat_recon + loss_adj_recon + loss_sg_recon +\
               sg_nce_loss + emb_loss + Cl_loss + kld_loss
        loss_dict = {'atom_feat_recon': loss_atom_feat_recon,
                     'adj_recon': loss_adj_recon,
                     'spacegroup_recon': loss_sg_recon,
                     'spacegroup_contrastive': sg_nce_loss,
                     'emb': emb_loss,
                     'contrastive': Cl_loss,
                     'kld': kld_loss
                     }
        # loss = y_loss
        # loss_dict = {'yloss':y_loss}
        return loss, loss_dict

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss


class DenoiseCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_encoder = Denoiser().to(cfg.DEVICE)
        self.target_encoder = crysAE().to(cfg.DEVICE)
        self.ntx = NTXentLoss()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, data):
        zin, pred_noise, mu, logvar = self.source_encoder(data)
        (edge_probs, adjs), atom_feat_p, sg_p, z3d_G, z3d = self.target_encoder(data)
        loss_atom_feat_recon = F.binary_cross_entropy_with_logits(atom_feat_p,
                                                                  data.node_features_t)

        emb_loss = self.mse(pred_noise, self.mse(zin, z3d)).mean()
        zdnoise = zin - pred_noise
        z_G = global_add_pool(zdnoise, data.atom_types_s_batch)
        z_G = F.normalize(z_G, dim=-1)
        pos_weight = torch.ones(cfg.MAX_ATOMS).to(cfg.DEVICE)
        pos_weight[0] = 0.1

        loss_adj_recon = F.nll_loss(edge_probs, adjs, weight=pos_weight)
        loss_sg_recon = F.nll_loss(sg_p, data.spacegroup_no_t)
        sg_nce_loss = self.ntx(z3d_G, data.spacegroup_t)
        sg_nce_loss += self.ntx(z_G, data.spacegroup_t)
        Cl_loss, Cl_acc = dual_CL(z_G, z3d_G)
        kld_loss = self.kld_loss(mu, logvar)

        loss = loss_atom_feat_recon + loss_adj_recon + loss_sg_recon + \
               sg_nce_loss + 10 * emb_loss + Cl_loss + 0.1 * kld_loss
        loss_dict = {'atom_feat_recon': loss_atom_feat_recon,
                     'adj_recon': loss_adj_recon,
                     'spacegroup_recon': loss_sg_recon,
                     'spacegroup_contrastive': sg_nce_loss,
                     'emb': emb_loss,
                     'contrastive': Cl_loss,
                     'kld': kld_loss
                     }
        return loss, loss_dict,

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss


class preTrainer:
    def __init__(self, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model = crysCL()
        self.device = cfg.DEVICE
        self.optim = optim.AdamW(params=self.model.parameters(),
                                 lr=cfg.TRAIN.lr)
        self.schedular = optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.TRAIN.max_epoch)
        self.min_valid_loss = 1e10
        self.best_epoch = 0
        self.directory = cfg.CHECKPOINT_DIR
        name_date = datetime.now().strftime("%m%d")
        self.exp_name = f"pretrain_{name_date}_" + cfg.DATASET_NAME.split('_')[-1]
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
            train_loss, train_loss_detail = self.eval_model(self.train_loader, 'train')
            valid_loss, valid_loss_detail = self.eval_model(self.valid_loader, 'valid')
            self.history['train'].append(train_loss)
            self.history['valid'].append(valid_loss)
            self.schedular.step()
            if valid_loss < self.min_valid_loss:
                torch.save(self.model.state_dict(), os.path.join(self.directory, self.exp_name))
                self.min_valid_loss = valid_loss
                self.best_epoch = epoch
                save_history(self.history, self.history_file)
            self.early_stopping(valid_loss)
            if self.early_stopping.early_stop:
                print(self.early_stopping.message)
                break

            if (epoch + 1) % cfg.TRAIN.snapshot_interval == 0:
                print(f'Epoch {epoch + 1}/{cfg.TRAIN.max_epoch}', end='  ')
                print(f'Train loss :{train_loss:.4f}', end='  ')
                print(f'Valid loss:{valid_loss:.4f}', end='  ')
                print(f'Interval time elapsed:{(time() - start_time) / 3600: .4f}')
                # print("Train loss details")
                # print(f'atom feat recon loss:{train_loss_detail["atom_feat_recon"]:.2f}', end = '  ')
                # print(f'adj recon loss:{train_loss_detail["adj_recon"]:.2f}', end = '  ')
                # print(f'space group loss:{train_loss_detail["spacegroup_recon"]:.2f}', end = '  ')
                # print(f'space group contrastive loss:{train_loss_detail["spacegroup_contrastive"]:.2f}', end = '  ')
                # print(f'2d 3d contrastive loss: {train_loss_detail["contrastive"]: .2f}', end = '  ')
                # print(f'input kld loss: {train_loss_detail["kld"]: .2f}', end = ' ')
                # print(f'source target embedding diff loss :{train_loss_detail["emb"]:.2f}')
        print("Training done")
        print(f"Best epoch :{self.best_epoch + 1}", end='  ')
        print(f'Best Valid loss:{self.min_valid_loss: .4f}')
        end_time = time()
        if epoch + 1 == cfg.TRAIN.max_epoch:
            self.model.load_state_dict(torch.load(os.path.join(self.directory, self.exp_name)))
            test_loss, test_loss_detail = self.eval_model(self.test_loader, 'test')
            self.history['test'].append(test_loss)
            save_history(self.history, self.history_file)
            print(f'Test Loss:{test_loss: .4f}')
            print("Test loss details")
            # print(f'atom feat recon loss:{test_loss_detail["atom_feat_recon"]:.2f}', end = '  ')
            # print(f'adj recon loss:{test_loss_detail["adj_recon"]:.2f}', end = '  ')
            # print(f'space group loss:{test_loss_detail["spacegroup_recon"]:.2f}', end = '  ')
            # print(f'space group contrastive loss:{test_loss_detail["spacegroup_contrastive"]:.2f}', end = '  ')
            # print(f'2d 3d contrastive loss: {test_loss_detail["contrastive"]: .2f}', end = '  ')
            # print(f'input kld loss: {test_loss_detail["kld"]: .2f}', end = ' ')
            # print(f'source target embedding diff loss :{test_loss_detail["emb"]:.2f}')
        print(f"Total time elapsed:{(end_time - start_time) / 3600: .4f}")
        self.print_result(self.test_loader)

    def eval_model(self, loader, split):
        if split == 'train':
            self.model.train()
            running_loss, Loss_dict = [], defaultdict(list)
            for batch in loader:
                batch = batch.to(cfg.DEVICE)
                loss, loss_dict = self.model(batch)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                running_loss.append(loss.item())
                for k, v in loss_dict.items():
                    Loss_dict[k].append(v.item())
            for k in Loss_dict:
                Loss_dict[k] = np.mean(Loss_dict[k])
            return np.mean(running_loss), Loss_dict
        else:
            self.model.eval()
            running_loss, Loss_dict = [], defaultdict(list)
            for batch in loader:
                batch = batch.to(cfg.DEVICE)
                with torch.no_grad():
                    loss, loss_dict = self.model(batch)
                    running_loss.append(loss.item())
                    for k, v in loss_dict.items():
                        Loss_dict[k].append(v.item())
            for k in Loss_dict:
                Loss_dict[k] = np.mean(Loss_dict[k])
            return np.mean(running_loss), Loss_dict

    def print_result(self, loader):
        self.model.eval()
        running_loss1, running_loss2, running_loss3 = [], [], []
        for batch in loader:
            batch = batch.to(cfg.DEVICE)
            with torch.no_grad():
                pred = self.model(batch)
                mae = self.mae_loss(pred, batch.y_t)
                mse = self.criterion(pred, batch.y_t)
                r2 = r2_score(batch.y_t.detach().cpu().numpy(), pred.detach().cpu().numpy())
                running_loss1.append(mae.item())
                running_loss2.append(mse.item())
                running_loss3.append(r2)
        Mae = np.mean(running_loss1)
        Mse = np.mean(running_loss2)
        Rmse = np.sqrt(Mse)
        R2 = np.mean(r2)
        print("Model Performance Metrics:")
        print(f"R2 Score: {R2:.4f} ")
        print(f"MAE: {Mae:.4f}")
        print(f"RMSE: {Rmse:.4f}")


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_CL(X, Y):
    X = F.normalize(X, dim=-1)
    Y = F.normalize(Y, dim=-1)

    criterion = nn.CrossEntropyLoss()
    B = X.size()[0]
    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, 0.1)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    # if args.CL_similarity_metric == 'EBM_dot_prod':
    #     criterion = nn.BCEWithLogitsLoss()
    #     neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)]
    #                        for i in range(args.CL_neg_samples)], dim=0)
    #     neg_X = X.repeat((args.CL_neg_samples, 1))
    #
    #     pred_pos = torch.sum(X * Y, dim=1) / args.T
    #     pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T
    #
    #     loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
    #     loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
    #     CL_loss = loss_pos + args.CL_neg_samples * loss_neg
    #
    #     CL_acc = (torch.sum(pred_pos > 0).float() +
    #               torch.sum(pred_neg < 0).float()) / \
    #              (len(pred_pos) + len(pred_neg))
    #     CL_acc = CL_acc.detach().cpu().item()

    return CL_loss, CL_acc


def dual_CL(X, Y):
    CL_loss_1, CL_acc_1 = do_CL(X, Y)
    CL_loss_2, CL_acc_2 = do_CL(Y, X)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2


if __name__ == "__main__":
    main()
