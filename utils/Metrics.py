import os
import csv
import torch
import numpy as np

class Metric:
    '''Metric computes accuracy/precision/recall/confusion_matrix with batch updates.'''

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y = []
        self.t = []

    def update(self, y, t):
        '''Update with batch outputs and labels.
        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].
        '''
        self.y.append(y)
        self.t.append(t)

    def _process(self, y, t):
        '''Compute TP, FP, FN, TN.
        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].
        Returns:
          (tensor): TP, FP, FN, TN, sized [num_classes,].
        '''

        tp = torch.empty(self.num_classes)
        for i in range(self.num_classes):
            ratio = y[:, i] / t[:, i]
            tp[i] = sum([1 for x in ratio if 0.9 <= float(x) <= 1.1])
        return tp

    def accuracy(self, reduction='mean'):
        '''Accuracy = (TP+TN) / (P+N).
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) accuracy.
        '''
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        total_len = t.shape[0]
        tp = self._process(y, t)
        if reduction == 'none':
            acc = tp / total_len
        else:
            acc = tp / total_len
        return acc



if __name__ == '__main__':
    test()