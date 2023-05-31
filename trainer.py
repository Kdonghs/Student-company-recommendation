import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer():
    def __init__(self, model, optimizer, train_loader, val_loader, scheduler, device,epochs,length,embedding_dim):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.length = length
        self.embedding_dim = embedding_dim
        self.criterion = nn.MSELoss().to(self.device)

    def fit(self, ):
        # li=[]
        self.model.to(self.device)
        best_score = 9999
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = []

            for x in tqdm(iter(self.train_loader)):
                torch.cuda.init()
                self.optimizer.zero_grad()
                _x,em,result = self.model(x,self.length,self.embedding_dim )
                loss = self.criterion(em, _x)

                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

            score = self.validation(self.model, em)
            print(f'Epoch : [{epoch}] Train loss : [{np.mean(train_loss)}] hitrate Score : [{score}])')
            # print(f'Epoch : [{epoch}] Train loss : [{np.mean(train_loss)}])')

            if best_score > np.mean(train_loss):
                best_score = np.mean(train_loss)
                torch.save(self.model.module.state_dict(), 'autoencoder_test.pth',
                           _use_new_zipfile_serialization=False)
                print(epoch, 'save')

    def validation(self, eval_model, result):
        eval_model.eval()
        total_loss = []
        p=0
        r=0

        with torch.no_grad():
            for x in tqdm(iter(self.val_loader)):
                _x,em,_result = self.model(x,self.length,self.embedding_dim)
                loss = self.criterion(em, _x)
                total_loss.append(loss.item())

        return np.mean(total_loss)