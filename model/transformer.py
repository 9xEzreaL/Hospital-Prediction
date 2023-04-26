import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0., pred=True):
        super().__init__()
        #out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.q = nn.Linear(in_features, in_features)
        self.k = nn.Linear(in_features, in_features)
        self.v = nn.Linear(in_features, in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.pred = pred
        if pred==True:
            self.fc2 = nn.Linear(hidden_features,1)
        else:
            self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x0 = x
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)
        attn = (q @ k.transpose(-2, -1))
        #print(attn.size())
        attn = attn.softmax(dim=-1)
        x = (attn @ v).squeeze(2)
        #print(x.size())q
        x += x0
        x1 = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.pred==False:
            x += x1

        x = x.squeeze(0)

        return x


class TF(nn.Module):
    def __init__(self, in_features, drop=0.):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=in_features - 2, embedding_dim=2)
        self.money_features = nn.Linear(1, 13)
        self.freq_features = nn.Linear(1, 13)

        self.mid_linear = nn.Linear(52, 64)

        self.Block1 = Mlp(in_features=52, hidden_features=128, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_2 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_3 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        self.Block_f = Mlp(in_features=77, hidden_features=128, act_layer=nn.GELU, drop=drop, pred=True)
        self.Block_m = Mlp(in_features=77, hidden_features=128, act_layer=nn.GELU, drop=drop, pred=True)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(1)

        x_f = F.softplus(self.embedding(x[:, :-2].long()))
        x_f = x_f.view(x_f.shape[0], -1)
        f_f = F.softplus(self.freq_features(x[:, -2:-1]))
        m_f = F.softplus(self.money_features(x[:, -1:]))

        mid_f = torch.cat([x_f, f_f, m_f], 1)
        mid_f = self.Block1(mid_f)
        # mid_f = torch.cat([mid_f, f_f, m_f], 1)
        mid_f_2 = F.softplus(self.mid_linear(mid_f)) # 64

        mid_f_f = torch.cat([mid_f_2, f_f], 1)
        freq = self.Block_f(mid_f_f)
        mid_m_f = torch.cat([mid_f_2, m_f], 1)
        spend = self.Block_m(mid_m_f)
        x = torch.cat([freq, spend], 1)
        return x

if __name__=="__main__":
    net = TF(in_features=15)
    a = torch.rand((5, 15))
    b = net(a)
