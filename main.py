import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import MyDataset
import model_co as auto_co
from model import Autoencoder
import prepro as pre
import trainer as Trainer
import os

class autoencoder:
    def __init__(self):
        self.EPOCHS = 100
        self.LR = 1e-6
        self.BS = 32
        self.SEED = 41
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.numworker = 20
        self.embedding_dim = 16

        # 데이터 불러오기
        self.data = pd.read_csv('', delimiter=',')

        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    def main(self):
        data,length = pre.prepro(self.data)
        print(len(data), ',', self.device)
        # 데이터프레임을 10개로 나누기

        trainList = [i[0] for i in pd.read_csv('').values.tolist()]
        testList = [i[0] for i in pd.read_csv('').values.tolist()]

        with open('') as f:
            notInTest = list(map(int, [i.replace('\n', '') for i in f.readlines()]))
        with open('') as f:
            notInTrain = list(map(int, [i.replace('\n', '') for i in f.readlines()]))

        for i in notInTest:
            testList.remove(i)
        for i in notInTrain:
            trainList.remove(i)

        train = data.loc[trainList]
        # train = data
        test = data.loc[testList]
        print(len(train),len(test))

        train_dataset = MyDataset.MyDataset(df=train, eval_mode=False)
        train_loader = DataLoader(train_dataset, batch_size=self.BS, shuffle=True, num_workers=self.numworker)

        val_dataset = MyDataset.MyDataset(df=test, eval_mode=True)
        val_loader = DataLoader(val_dataset, batch_size=self.BS, shuffle=False, num_workers=self.numworker)

        model = nn.DataParallel(Autoencoder(self.embedding_dim, length))
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10,
                                                               threshold_mode='abs', min_lr=1e-5, verbose=True)
        trainer = Trainer.Trainer(model, optimizer, train_loader, val_loader, scheduler, self.device, self.EPOCHS, length,self.embedding_dim)
        trainer.fit()

if __name__ == '__main__':
    autoencoder().main()


