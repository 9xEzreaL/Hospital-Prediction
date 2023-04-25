import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    def __init__(self, num_feature, num_classes=2):
        super(RNN, self).__init__()

        if num_feature > 2:
            self.embedding = nn.Embedding(num_embeddings=num_feature-2, embedding_dim=2)
        self.money_features = nn.Linear(1, 13)
        self.freq_features = nn.Linear(1, 13)
        self.mid_linear = nn.Linear(52, 115)
        # self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
        #     input_size=48,
        #     hidden_size=128,         # rnn hidden unit
        #     num_layers=2,           # number of rnn layer
        #     batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        # )

        self.out_freq = nn.Linear(128, num_classes//2)
        self.out_spend = nn.Linear(128, num_classes//2)
        # self.actf = nn.softplus()

        self.num_feature = num_feature

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(1)
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        if self.num_feature > 2:
            x_f = F.softplus(self.embedding(x[:, :-2].long()))
            x_f = x_f.view(x_f.shape[0], -1)

        f_f = F.softplus(self.freq_features(x[:, -2:-1]))
        m_f = F.softplus(self.money_features(x[:, -1:]))

        mid_f = torch.cat([x_f, f_f, m_f], 1)
        mid_f_2 = F.softplus(self.mid_linear(mid_f))


        out_freq = self.out_freq(torch.cat([f_f, mid_f_2], 1))
        out_spend = self.out_spend(torch.cat([m_f, mid_f_2], 1))

        # r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        # out_money = self.out_money(r_out[:, -1, :])
        # out_spend = self.out_spend(r_out[:, -1, :])
        out = torch.cat([out_freq, out_spend], 1)
        return out

if __name__=="__main__":
    model = RNN(10)
    a = torch.rand((20, 1, 10))
    print(model)
    b = model(a)
    print(b.shape)