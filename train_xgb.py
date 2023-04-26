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
from torchsampler import ImbalancedDatasetSampler
import torch.optim as optim
from utils.configurations import root, meta_csv, save_to
from sklearn.metrics import mean_squared_error

from model.xgb_dnn import XGB

"""
Train a classfier model and save it so that you can use this classfier to test your generated data.
"""


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', dest='img_size', type=int, default=256)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='# of epochs')
    parser.add_argument('--bs_per_gpu', dest='batch_size_per_gpu', type=int, default=20)  # training batch size
    parser.add_argument('--lr', dest='lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--net', dest='net', default='rnn')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--exp', dest='experiment_name',
                        default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    parser.add_argument('--features', dest='features', type=list, default=['age_65', 'Emergency', 'hypertension', 'DM',
                                                                           'hyperlipidemia', 'CKD', 'chronic_hepatitis',
                                                                            'COPD_asthma', 'gout_hyperua', 'heart_disease',
                                                                           'CVD', 'ulcers', 'Cancer_metastatic', 'freq', 'appl_dot_sum'])
    parser.add_argument('--classes', dest='classes', type=list, default=['freq', 'spend_16'])
    return parser.parse_args(args)



if __name__ == '__main__':
    args = parse()

    args.lr_base = args.lr

    os.makedirs(join(save_to, args.experiment_name), exist_ok=True)
    os.makedirs(join(save_to, args.experiment_name, 'checkpoint'), exist_ok=True)
    writer = SummaryWriter(join(save_to, args.experiment_name, 'summary'))

    with open(join(save_to, args.experiment_name, 'setting.txt'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

    num_gpu = torch.cuda.device_count()
    progressbar = Progressbar()

    from data.dataloader import VanillaDataset

    train_dataset = VanillaDataset(root, meta_csv, mode='train', features=args.features)
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=6101,
                                       num_workers=10, drop_last=True, shuffle=True)

    eval_dataset = VanillaDataset(root, meta_csv, mode='eval', features=args.features)
    eval_dataloader = data.DataLoader(dataset=eval_dataset, batch_size=1492,
                                      num_workers=10, drop_last=True, shuffle=False)

    print('Training images:', len(train_dataset))
    print('Eval images:', len(eval_dataset))
    import xgboost as xgb
    xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                               max_depth=5, alpha=10, n_estimators=10)
    for img, label, id in progressbar(train_dataloader):
        img = np.array(img.squeeze(1))
        label = label[:,0]

        xg_reg.fit(img, label)
    for img, label, id in progressbar(eval_dataloader):
        img = np.array(img.squeeze(1))
        preds = xg_reg.predict(img)
    ratio = preds / label[:, 0]
    all = sum([1 for x in ratio if 0.9 <= float(x) <= 1.1])
    print('acc:', all/len(preds))
    rmse = np.sqrt(mean_squared_error(label[:,0], preds))
    print(rmse)
