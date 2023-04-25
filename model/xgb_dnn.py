import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import xgboost as xgb

class XGB(nn.Module):
    def __init__(self, num_feature=15, num_classes=2):
        super(XGB, self).__init__()

        self.xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                                  max_depth = 5, alpha = 10, n_estimators = 10)
        print(self.xg_reg)
        # if num_feature > 2:
        #     self.embedding = nn.Embedding(num_embeddings=num_feature-2, embedding_dim=2)
        # self.money_features = nn.Linear(1, 13)
        # self.freq_features = nn.Linear(1, 13)
        # self.mid_linear = nn.Linear(52, 115)
        # self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
        #     input_size=48,
        #     hidden_size=128,         # rnn hidden unit
        #     num_layers=2,           # number of rnn layer
        #     batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        # )

        # self.out_freq = nn.Linear(128, num_classes//2)
        # self.out_spend = nn.Linear(128, num_classes//2)
        # self.actf = nn.LeakyReLU()
        #
        # self.num_feature = num_feature

    def forward(self, x):
        pass
        # return out

if __name__=='__main__':
    model =  XGB()
    print(model)