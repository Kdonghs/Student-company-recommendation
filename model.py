import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, embedding_dim,length):
        super(Autoencoder, self).__init__()

        # 임베딩 테이블 생성
        # 15554 is max grup_cd
        self.embedding = nn.Embedding(15554, embedding_dim,max_norm=True)

        self.result = nn.Linear(in_features=32, out_features=21121)

        self.encoder = nn.Sequential(
            nn.Linear(in_features=6*length*embedding_dim, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=6*length*embedding_dim,),
            nn.Tanh()
        )

    def forward(self, x,length,embedding_dim):
        batch_size = len(x[0])

        # 입력 데이터 변형 및 타입 변환
        x = torch.stack(x, dim=0)
        x = x.view(batch_size, -1)

        # 임베딩 적용
        em = self.embedding(x)
        # print(em.size())

        # 차원 축소
        x = em.view(batch_size, -1)

        # 인코더
        x = self.encoder(x)

        result = self.result(x)

        # 디코더
        x = self.decoder(x)

        return x.view(batch_size, 6*length,embedding_dim),em, result

class AutoencoderTest(nn.Module):
    def __init__(self, embedding_dim,length):
        super(AutoencoderTest, self).__init__()

        # 임베딩 테이블 생성
        # 15554 is max grup_cd
        self.embedding = nn.Embedding(15554, embedding_dim,max_norm=True)

        self.result = nn.Linear(in_features=32, out_features=21121)

        self.encoder = nn.Sequential(
            nn.Linear(in_features=6*length*embedding_dim, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=6*length*embedding_dim,),
            nn.Tanh()
        )

    def forward(self, x,length,embedding_dim):
        batch_size = 1

        # 입력 데이터 변형 및 타입 변환
        x = torch.stack(x, dim=0)
        x = x.view(batch_size, -1)

        # 임베딩 적용
        em = self.embedding(x)
        # print(em.size())

        # 차원 축소
        x = em.view(batch_size, -1)

        # 인코더
        x = self.encoder(x)

        result = self.result(x)

        # 디코더
        x = self.decoder(x)

        return x.view(batch_size, 6*length,embedding_dim),em, result
