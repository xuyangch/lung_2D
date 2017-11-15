# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import data
import matplotlib
from constants import *
import random
from test import get_roc_curve

# calculate entropy
def entropy(s, j, l):
    assert (j == l)
    p = np.array([s[j],1-s[j]])
    entr = 0.0
    entr += (-(p*np.log(p))).sum()
    return entr

# calculate diversity
def diversity(s, j, l):
    pj = np.array([s[j], 1 - s[j]])
    pl = np.array([s[l], 1 - s[l]])
    diver = 0.0
    diver += ((pj-pl) * np.log(pj/pl)).sum()
    return diver

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0
    for epoch in range(7 * num_epochs + 15):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-'*10)

        #Each epoch has a trainning and validate process
        for phase in ['aift', 'val', 'test']:
            print('phase: '+phase)

            # use AIFT to select labeled samples
            if (phase == 'aift'):

                # init
                model.train(True)
                dsets['aift'].show_full_label = True
                R = np.zeros([len(dsets['aift']), int(ALPHA * PATCH_NUM), int(ALPHA * PATCH_NUM)])
                scores = []
                candidates_need_label = []

                # iterate over all candidates
                for i in range(len(dsets['aift'])):

                    # check whether the sample is labeled
                    if dsets['aift'].has_index_from_num(i):
                        continue
                    candidates_need_label.append(i)

                random.shuffle(candidates_need_label)
                candidates_need_label = candidates_need_label[:BETA]
                dsets['aift'].add_labeled_candidate(candidates_need_label)

            else:
                model.train(False)

            # set use_volatile
            use_volatile = ((phase == 'val') or (phase == 'test'))

            # use limited label when trainning
            dsets['aift'].show_full_label = False

            # init
            running_loss = 0.0
            running_corrects = 0
            predictions = []
            label_predictions = []
            Y_labels = []

            # Iterate over data
            for data in dset_loaders[phase]:

                # get the inputs
                inputs, labels = data

                if phase == 'test':
                    inputs = inputs.view(-1,3,224,224)
                    # print(inputs.shape)

                # wrap them in variable
                if use_gpu:
                    inputs, labels = Variable(inputs, volatile=use_volatile).cuda(), \
                                     Variable(labels, volatile=use_volatile).cuda()
                else:
                    inputs, labels = Variable(inputs, volatile=use_volatile), \
                                     Variable(labels, volatile=use_volatile)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                # in test phase, we get all the augmented patch
                if phase == 'test':

                    # get the mean prob. of all patches
                    outputs = outputs.view(-1, 2)
                    outputs = nn.Softmax()(outputs)
                    outputs = outputs.view(-1, 12, 2)
                    outputs = torch.mean(outputs, 1)
                    _, preds = torch.max(outputs.data, 1)

                    # statistics
                    predictions += outputs.data.tolist()
                    label_predictions += preds.tolist()
                    Y_labels += labels.data.tolist()
                    running_corrects += torch.sum(preds == labels.data)

                # other phase, simple routine
                else:
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in trainning/aift phase
                    if phase == 'aift':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds == labels.data)
                    predictions += (nn.Softmax()(outputs)).data.tolist()
                    label_predictions += preds.tolist()
                    Y_labels += labels.data.tolist()

            # data enumaration ends, calc data
            epoch_loss = running_loss / len(dsets[phase])
            epoch_acc = running_corrects / len(dsets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # transform to np
            predictions = np.asarray(predictions)
            label_predictions = np.asarray(label_predictions)
            Y_labels = np.asarray(Y_labels)

            # calc fp, tp, auc
            fpr, tpr, roc_auc = get_roc_curve(Y_labels, predictions)
            print('AUC is: ' + str(roc_auc))

            if (phase == 'aift'):
                print('%d labels used', len(dsets['aift']))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
                torch.save(model_ft.state_dict(), 'weights_aift.save')
                print('weights saved')

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr = 0.001, lr_decay_epoch = 10):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

# Data augmentation and normalization for training
# Just normalization for validation

data_transforms = {
    'aift': transforms.Compose([
        transforms.Scale((224,224)),
        data.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Scale((224,224)),
        data.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Scale((224,224)),
        data.ToTensor()
    ]),
}

dsets = {x : data.Luna2DData(x, data_transforms[x])
         for x in ['aift', 'val', 'test']}

dset_loaders = {x: DataLoader(dsets[x], batch_size=96,
                                               shuffle=True, num_workers=8)
                for x in ['aift','val']}
dset_loaders['test'] = DataLoader(dsets['test'], batch_size=8,
                                               shuffle=True, num_workers=8)

dset_sizes = {x: len(dsets[x]) for x in ['aift','val','test']}
dset_classes = dsets['aift'].classes

use_gpu = torch.cuda.is_available()

# model_ft = models.resnet18(pretrained=True)
model_ft = models.alexnet(pretrained=True)

# disable grads of all parameters
for param in model_ft.features.parameters():
    param.requires_grad = False

# replace final layer
mod = list(model_ft.classifier.children())
mod.pop()
mod.append(torch.nn.Linear(4096,2))
new_classifier = torch.nn.Sequential(*mod)
model_ft.classifier = new_classifier




if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr = 0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.classifier.parameters())
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=400)
# torch.save(model_ft.state_dict(), 'weights.save')