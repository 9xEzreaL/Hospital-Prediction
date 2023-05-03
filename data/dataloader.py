import os
import glob
import math
import random
import torch
import pandas as pd
import numpy as np
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# from utils.configurations import root, meta_csv, save_to


class VanillaDataset(Dataset):
    def __init__(self,
                 root,
                 meta_csv,
                 mode,
                 features=['age_65', 'Emergency', 'hypertension', 'DM', 'hyperlipidemia', 'CKD', 'chronic_hepatitis',
                           'COPD_asthma', 'gout_hyperua', 'heart_disease', 'CVD', 'ulcers', 'Cancer_metastatic', 'freq', 'appl_dot_sum'],
                 objective=['freq', 'spend_16']
                 ):

        meta_info = pd.read_csv(os.path.join(root, meta_csv))
        if mode == 'train':
            self.meta_info = meta_info.loc[700:6800]
        elif mode == 'eval':
            self.meta_info = pd.concat([meta_info.loc[:700], meta_info.loc[6800:]], 0)

        self.features = features
        self.objective = objective
        self.mode = mode

    def __getitem__(self, index):
        id = self.meta_info.iloc[index]['id']

        feature_list = [self.meta_info.iloc[index][i] for i in self.features]
        features = torch.Tensor(feature_list).unsqueeze(0)
        if int(self.meta_info.iloc[index]['appl_dot_sum']) > 0:
            feature_list[-1] = math.log(int(self.meta_info.iloc[index]['appl_dot_sum']))
        else:
            tmp_sum = 1
            feature_list[-1] = math.log(tmp_sum)

        label_freq = self.meta_info.iloc[index][self.objective[0]]
        if int(self.meta_info.iloc[index][self.objective[1]]) > 0:
            label_money = math.log(int(self.meta_info.iloc[index][self.objective[1]]))
        else:
            label_money = 1
            label_money = math.log(label_money)

        label = torch.Tensor([label_freq, label_money])

        return features, label, id

    def __len__(self):
        return len(self.meta_info['id'])

if __name__=="__main__":
   csv = '/media/ExtHDD01/Dataset/ktong/子毅0417/df_final_201516_20230419.csv'
   csv_15 = '/media/ExtHDD01/Dataset/ktong/子毅0417/df_final_15_18_0419.csv'
   csv_16 = '/media/ExtHDD01/Dataset/ktong/子毅0417/df_final_16_18_0419.csv'
   meta_info = pd.read_csv(csv)
   meta_info_15 = pd.read_csv(csv_15)
   meta_info_16 = pd.read_csv(csv_16)
   find_id = meta_info_15["id"].iloc[0]
   print(meta_info_16[meta_info_16["id"]==find_id])
   # print(meta_info.loc[:600].shape)
   # print(meta_info.loc[7000:].shape)
   # test = pd.concat([meta_info.loc[:700], meta_info.loc[6800:]], 0)
   # print(test.shape)
