"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from ..meter import AverageMeter
from ..metric import binary_accuracy

import os, sys
import torch.nn as nn

import torch
import pandas as pd
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# In[213]:


import numpy as np
from sklearn import svm


# Comment out sigmoid to simulate svm https://github.com/kazuto1011/svm-pytorch/blob/master/main.py
# but svc needs sigmoid
class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x


def calculate(source_feature: torch.Tensor, target_feature: torch.Tensor,
              device, progress=True, training_epochs=50):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.

    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`\mathcal{A}`-distance
    """
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    feature = torch.cat([source_feature, target_feature], dim=0)
    label = torch.cat([source_label, target_label], dim=0)

    dataset = TensorDataset(feature, label)
    length = len(dataset)
    # train_size = int(0.8 * length)
    # similar to half_source
    train_size = int(0.5 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=len(dataset), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(dataset), shuffle=False)

    # anet = ANet(feature.shape[1]).to(device)
    # optimizer = SGD(anet.parameters(), lr=0.01)
    # a_distance = 2.0
    # for epoch in range(training_epochs):
    #     anet.train()
    #     for (x, label) in train_loader:
    #         x = x.to(device)
    #         label = label.to(device)
    #         anet.zero_grad()
    #         y = anet(x)
    #         loss = F.binary_cross_entropy(y, label)
    #         loss.backward()
    #         optimizer.step()

    #     anet.eval()
    #     meter = AverageMeter("accuracy", ":4.2f")
    #     with torch.no_grad():
    #         for (x, label) in val_loader:
    #             x = x.to(device)
    #             label = label.to(device)
    #             y = anet(x)
    #             acc = binary_accuracy(y, label)
    #             meter.update(acc, x.shape[0])
    #     error = 1 - meter.avg / 100

    #     # Similar to if test_risk > .5: test_risk = 1. - test_risk
    #     if error > .5:
    #         error = 1. - error

    #     a_distance = 2 * (1 - 2 * error)
    #     if progress:
    #         print("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance))

    # New code
    for (train_X, train_Y) in train_loader:
        print('trainX, trainY')
        for (test_X, test_Y) in val_loader:
            print('testX, testY')
            best_risk = 1.0

            train_Y = train_Y.flatten() 
            test_Y = test_Y.flatten()

            train_X = train_X.tolist()
            train_Y = train_Y.tolist()
            test_X = test_X.tolist()
            test_Y = test_Y.tolist()

            # df = pd.DataFrame([train_X, train_Y])
            # df.to_csv('a_distance_test.csv')

            clf = svm.SVC(kernel='linear', verbose=False)
            clf.fit(train_X, train_Y)

            train_risk = np.mean(clf.predict(train_X) != train_Y)
            test_risk = np.mean(clf.predict(test_X) != test_Y)

            print('train risk: %f  test risk: %f' % (train_risk, test_risk))

            if test_risk > .5:
                test_risk = 1. - test_risk

            best_risk = min(best_risk, test_risk)
            temp = 2 * (1. - 2 * best_risk)
            print('best risk: %f temp a-distance: %f ' % (best_risk,temp))

            a_distance = 2 * (1. - 2 * best_risk)

    return a_distance


# Fatima's code
def proxy_a_distance(source_X, target_X, verbose=True):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]


    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source[0:half_source, :], target[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source[half_source:, :], target[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0

    clf = svm.SVC(kernel='linear', verbose=False)
    clf.fit(train_X, train_Y)

    train_risk = np.mean(clf.predict(train_X) != train_Y)
    test_risk = np.mean(clf.predict(test_X) != test_Y)

    print('train risk: %f  test risk: %f' % (train_risk, test_risk))

    if test_risk > .5:
        test_risk = 1. - test_risk

    best_risk = min(best_risk, test_risk)
    temp = 2 * (1. - 2 * best_risk)
    print('best risk: %f temp a-distance: %f ' % (best_risk,temp))

    return 2 * (1. - 2 * best_risk)