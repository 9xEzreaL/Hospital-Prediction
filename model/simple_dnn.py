import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(torch.nn.Module):
    def __init__(self, n_feature=15, n_hidden=64, n_output=1):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        # self.predict_freq = torch.nn.Linear(n_hidden, n_output)   # output layer
        self.predict_money = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = x.squeeze(1)
        # x = F.relu(self.hidden(x))
        # activation function for hidden layer
        # freq = self.predict_freq(x)             # linear output
        x = F.softplus(self.hidden(x))
        money = self.predict_money(x)

        # out = torch.cat([freq, money], 1)
        out = money
        return out

if __name__=='__main__':
    model = Net(15, 64, 2)
    a = torch.rand((10, 1, 15))
    b = model(a)
    print(b.shape)