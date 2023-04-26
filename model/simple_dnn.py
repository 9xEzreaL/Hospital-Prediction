import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DNN(nn.Module):
    def __init__(self, num_feature, num_classes=2):
        super(DNN, self).__init__()

        if num_feature > 2:
            self.embedding = nn.Embedding(num_embeddings=num_feature-2, embedding_dim=2)
        self.money_features = nn.Linear(1, 13)
        self.freq_features = nn.Linear(1, 13)
        self.mid_linear = nn.Linear(52, 115)

        self.out_freq = nn.Linear(128, num_classes//2)
        self.out_spend = nn.Linear(128, num_classes//2)

        self.num_feature = num_feature

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(1)

        if self.num_feature > 2:
            x_f = F.softplus(self.embedding(x[:, :-2].long()))
            x_f = x_f.view(x_f.shape[0], -1)

        f_f = F.softplus(self.freq_features(x[:, -2:-1]))
        m_f = F.softplus(self.money_features(x[:, -1:]))

        mid_f = torch.cat([x_f, f_f, m_f], 1)
        mid_f_2 = F.softplus(self.mid_linear(mid_f))


        out_freq = self.out_freq(torch.cat([f_f, mid_f_2], 1))
        out_spend = self.out_spend(torch.cat([m_f, mid_f_2], 1))

        out = torch.cat([out_freq, out_spend], 1)
        return out