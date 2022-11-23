import torch
import torch.optim as optim
import torch.nn as nn
import os, pickle
from time import time
from tqdm import tqdm
from model import LatticeMLP, FAN
import numpy as np
import matplotlib.pyplot as plt

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

class ModelTrainer:
    def __init__(self, args, directory, train_loader, valid_loader, test_loader, test_indices):
        self.args = args
        self.directory = directory
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.test_indices = test_indices
        self.exp_name = args.exp_name
        self.device = args.device
        self.len_data = args.num_train
        self.model = FAN(args).to(self.device)
        self.cosine = nn.CosineSimilarity()
        self.mse = nn.MSELoss()
        self.optimizer = optim.AdamW(params=self.model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)

        if args.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer)
        elif args.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5)
        elif args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)
        self.early_stopping = EarlyStopping(args.patience)

        self.history = {'train': [], 'valid': [], 'test': []}
        self.history['test_indices'] = test_indices
        self.history_file = os.path.join(directory, 'history_' + self.args.exp_name + '.pickle')
        self.min_train_loss = 1e10
        self.min_valid_loss = 1e10
        self.best_epoch = 0

    def train(self):
        # get initial test results
        print("start pre-training!")
        start_time = time()
        # start training
        for epoch in range(self.args.epochs):
            tmp_train_loss = self.evaluate_model(self.train_loader, 'train')
            self.history['train'].append(np.average(tmp_train_loss))
            tmp_valid_loss = self.evaluate_model(self.valid_loader, 'valid')
            self.history['valid'].append(np.average(tmp_valid_loss))

            print(f'Epoch {epoch + 1}/{self.args.epochs}')
            print(f"\tTrain loss: {self.history['train'][-1]:.5f}")
            print(f"\tValid loss: {self.history['valid'][-1]:.5f}")
            print(f'\tTime elapsed: {(time() - start_time):.3f}')

            self.scheduler_step(epoch, self.history['train'][-1], self.history['valid'][-1])
            if self.early_stopping.early_stop:
                print(self.early_stopping.message)
                break

        print("Training Done!")
        print(f'Best result at epoch: {self.best_epoch + 1}, '
              f'Min Train loss: {self.min_train_loss:.5f}, '
              f'Min Valid loss: {self.min_valid_loss:.5f}')
        end_time = time()
        self.plot_results()
        # Test
        if epoch != 0:
            self.model.load_state_dict(torch.load(os.path.join(self.directory, self.args.exp_name)))
            tmp_test_loss = self.evaluate_model(self.test_loader, 'test')
            self.history['test'].append(np.average(tmp_test_loss))
            save_history(self.history, self.history_file)
            print(f"\tTest loss: {self.history['test'][-1]:.5f}")
        print(f'Total Time elapsed: {(end_time - start_time):.3f}')
        with torch.no_grad():
            test_sample = self.test_loader.dataset[-1]
            fg, lbl = test_sample
            fg = fg.to(self.device)
            lbl = torch.tensor(lbl).to(self.device)
            out = self.model(fg)
            print(f"pred matrix\n{out}\ntrue matrix\n{lbl}")
            print(f"cosine sim {self.cosine(out,lbl).item():.5f}")
            print(f"mse {self.mse(out.squeeze(),lbl).item():.5f}")

    def scheduler_step(self, epoch, train_loss, valid_loss):
        if self.args.scheduler == 'plateau':
            self.scheduler.step(valid_loss)
        else:
            self.scheduler.step()

        if valid_loss < self.min_valid_loss:
            torch.save(self.model.state_dict(), os.path.join(self.directory, self.args.exp_name))
            self.min_train_loss = train_loss
            self.min_valid_loss = valid_loss
            self.best_epoch = epoch
            save_history(self.history, self.history_file)

        self.early_stopping(valid_loss)

    def evaluate_model(self, loader, split):
        if split == 'train':
            running_loss = []
            self.model.train()
            for fg, label in tqdm(loader):
                label = label.to(self.device)
                loss = self.calculate_loss(fg, label, split)
                running_loss.append(loss)
            return running_loss

        else:
            running_loss = []
            self.model.eval()
            for fg, label in tqdm(loader):
                label = label.to(self.device)
                with torch.no_grad():
                    loss = self.calculate_loss(fg, label, split)
                    running_loss.append(loss)
            return running_loss

    def calculate_loss(self, fg, label, split):
        fg = fg.to(self.device)
        out = self.model(fg)
        cos = self.cosine(out,label)
        cos_loss = 1-torch.mean(cos)
        mse_loss = self.mse(out,label.float())
        loss = cos_loss + mse_loss
        if split == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def plot_results(self):
        plt.plot(range(len(self.history['train'])), self.history['train'], label='train')
        plt.plot(range(len(self.history['valid'])), self.history['valid'], label='valid')
        plt.legend()
        plt.savefig(os.path.join(self.directory, self.args.exp_name + ".png"))

