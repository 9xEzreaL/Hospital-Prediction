import torch
import torch.nn as nn
import os
import time
import argparse
import datetime
import json
from os.path import join
import numpy as np
import torch.utils.data as data
from utils.Metrics import Metric
from utils.helpers import Progressbar, add_scalar_dict
from tensorboardX import SummaryWriter
import torch.optim as optim
from utils.configurations import root, meta_csv, save_to

from model.RNN import RNN
from model.simple_dnn import DNN
from model.transformer import TF

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', dest='img_size', type=int, default=256)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='# of epochs')
    parser.add_argument('--bs_per_gpu', dest='batch_size_per_gpu', type=int, default=20)  # training batch size
    parser.add_argument('--lr', dest='lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--net', dest='net', default='dnn')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--exp', dest='experiment_name',
                        default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    parser.add_argument('--features', dest='features', type=list, default=['age_65', 'Emergency', 'hypertension', 'DM',
                                                                           'hyperlipidemia', 'CKD', 'chronic_hepatitis',
                                                                            'COPD_asthma', 'gout_hyperua', 'heart_disease',
                                                                           'CVD', 'ulcers', 'Cancer_metastatic', 'freq', 'appl_dot_sum'])
    parser.add_argument('--classes', dest='classes', type=list, default=['freq', 'spend_16']) # Objectiive
    return parser.parse_args(args)


class Classifier:
    def __init__(self, args, net):
        self.args = args
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.model = self.network_map(net)(len(args.features))
        self.model.train()
        self.model.cuda()

        if self.multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.optim_model = optim.Adam(self.model.parameters(), lr=args.lr)
        self.it = 0

    def set_lr(self, lr):
        for g in self.optim_model.param_groups:
            g['lr'] = lr

    def train_model(self, img, label, metric):  # (self, img, label) [0., 0., 0., 0., 1., 0.]
        for p in self.model.parameters():
            p.requires_grad = True

        pred = self.model(img)
        label = label.float()

        loss_freq = nn.MSELoss()(pred[:, 0], label[:, 0])
        loss_money = nn.MSELoss()(pred[:, 1], label[:, 1])

        loss = loss_freq.mean() + loss_money.mean()

        loss.backward()
        self.optim_model.step()
        self.optim_model.zero_grad()

        metric.update(pred, label)
        acc = metric.accuracy()

        errD = {
            'freq_loss': loss_freq.mean().item(),
            'money_loss': loss_money.mean().item()
        }
        self.it += 1
        return errD, acc

    def eval_model(self, img, label, metric):  # (self, img, label) [0., 0., 0., 0., 1., 0.]
        with torch.no_grad():
            pred = self.model(img)
        label = label.type(torch.float)

        metric.update(pred, label)
        acc = metric.accuracy()

        return acc

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path):
        states = {
            'model': self.model.state_dict(),
            'optim_model': self.optim_model.state_dict(),
        }
        torch.save(states, path)

    def load(self, fine_tune=False, ckpt=None):
        if fine_tune:
            states = torch.load(ckpt)
            self.model.load_state_dict(states['model'])
            for module in self.model.modules():
                # print(module)
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()

    def network_map(self, net):
        network_mapping = {
            'rnn' : RNN,
            'dnn' : DNN,
            'attn' : TF,
        }
        return network_mapping[net]


if __name__ == '__main__':
    args = parse()

    args.lr_base = args.lr

    os.makedirs(join(save_to, args.experiment_name), exist_ok=True)
    os.makedirs(join(save_to, args.experiment_name, 'checkpoint'), exist_ok=True)
    writer = SummaryWriter(join(save_to, args.experiment_name, 'summary'))

    with open(join(save_to, args.experiment_name, 'setting.txt'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

    num_gpu = torch.cuda.device_count()

    classifier = Classifier(args, net=args.net)
    progressbar = Progressbar()

    from data.dataloader import VanillaDataset

    train_dataset = VanillaDataset(root, meta_csv, mode='train', features=args.features)
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size_per_gpu * num_gpu,
                                       num_workers=10, drop_last=True, shuffle=True)

    eval_dataset = VanillaDataset(root, meta_csv, mode='eval', features=args.features)
    eval_dataloader = data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size_per_gpu * num_gpu,
                                      num_workers=10, drop_last=True, shuffle=False)

    print('Training images:', len(train_dataset))
    print('Eval images:', len(eval_dataset))

    it = 0
    it_per_epoch = len(train_dataset) // (args.batch_size_per_gpu * num_gpu)
    for epoch in range(args.epochs):
        lr = args.lr_base * (0.9 ** epoch)
        classifier.set_lr(lr)
        classifier.train()
        writer.add_scalar('LR/learning_rate', lr, it + 1)
        metric_tr = Metric(num_classes=len(args.classes))
        metric_ev = Metric(num_classes=len(args.classes))
        for img, label, id in progressbar(train_dataloader):
            img = img.cuda()
            label = label.cuda()

            label = label.type(torch.float)
            img = img.type(torch.float)

            errD, acc = classifier.train_model(img, label, metric_tr)
            it += 1
            progressbar.say(epoch=epoch, freq_loss=errD['freq_loss'], money_loss=errD['money_loss'], acc=acc.detach().numpy())
            # progressbar.say(epoch=epoch, freq_loss=errD['freq_loss'], acc=acc.detach().numpy())

        classifier.eval()
        for img, label, id in progressbar(eval_dataloader):
            img = img.cuda()
            label = label.cuda()

            img = img.type(torch.float)
            label = label.type(torch.float)

            acc = classifier.eval_model(img, label, metric_ev)
            it += 1
            progressbar.say(epoch=epoch, acc=acc.detach().numpy())


        classifier.save(os.path.join(
            save_to, args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
        ))
