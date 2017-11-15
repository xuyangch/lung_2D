from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
import matplotlib
import sys
from constants import *
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), self.interpolation)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), self.interpolation)

class RandCrop(object):
    def __call__(self, img):

        # random crop
        k = np.random.randint(3)
        w, h = img.size
        th = tw = 70 - 10 * k
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

# Rotate 90 random times
class RandRotate(object):
    def __call__(self, img):

        # random rotate
        k = np.random.randint(4)
        img = img.rotate(90 * k)

        return img

class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backard compability
            return img.float().div(255)
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if img.shape != torch.Size([1,224,224]):

            print(str(img.shape))
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

class Luna2DData(Dataset):
    """Luna2D dataset."""

    def __init__(self, phase, transform=None):
        self.num_to_index = []
        self.data = []
        self.label = []
        self.phase = phase
        self.transform = transform
        self.aift_labeled = []
        self.classes = {'Not', 'Yes'}
        self.show_full_label = True

        # aift uses the same data as train, but with no upsample
        if phase == 'aift':
            phase = 'train'

        data = pd.read_pickle(phase + 'data')
        label = pd.read_pickle(phase + 'labels')
        self.ori_label = label

        for i in range(len(label.index)):
            idx = data.index[i]
            self.num_to_index.append(idx)
            self.data.append(data.loc[idx])
            self.label.append(label.loc[idx])

        # upsample positive only if training
        if self.phase == 'train':
            for i in range(len(self.label)):
                if self.label[i] == 1:
                    self.label.extend(ADD_AUG*[1])
                    self.data.extend(ADD_AUG*[self.data[i]])
                    self.num_to_index.extend(ADD_AUG*[self.num_to_index[i]])

        # calc pos&neg number
        pos, neg = 0, 0
        for i in range(len(self.num_to_index)):
            if self.label[i] == 1:
                pos += 1
            elif self.label[i] == 0:
                    neg += 1
            else:
                print('error!')
        print('positive: %d\nnegative: %d\ntotal: %d\n' % (pos,neg,pos+neg))

    def __len__(self):
        if self.phase == 'aift':
            if self.show_full_label:
                return len(self.data)
            else:
                return len(self.aift_labeled)
        else:
            return len(self.data)

    def __getitem__(self, i):

        # used to determine directory
        phase = self.phase
        if phase == 'aift':
            phase = 'train'

        # get the index in Dataframe
        if (self.phase == 'aift') and (self.show_full_label == False):
            idx = self.aift_labeled[i]
            label = self.ori_label.loc[idx]
        else:
            idx = self.num_to_index[i]
            label = self.label[i]

        # get image name
        if sys.platform == 'darwin':
            img_name = os.path.join('local', phase, 'image_' + str(idx) + '.jpg')
        else:
            img_name = os.path.join(phase, 'image_'+str(idx)+'.jpg')

        label = int(label)
        image = Image.open(img_name)

        # get all augmented pictures when aift selecting
        if ((self.phase == 'aift') and (self.show_full_label == True)) or (self.phase == 'test'):
            img_aug = []
            w, h = image.size

            # append different scale
            for k in range(3):
                th = tw = 70 - 10 * k
                x1 = int(round((w - tw) / 2.))
                y1 = int(round((h - th) / 2.))
                img_aug.append(image.crop((x1, y1, x1 + tw, y1 + th)))

            # append different rotation
            for k in range(3):
                img_aug.append(img_aug[k].rotate(90))
                img_aug.append(img_aug[k].rotate(180))
                img_aug.append(img_aug[k].rotate(270))

            if self.transform:
                for k in range(len(img_aug)):
                    img_aug[k] = self.transform(img_aug[k])
                    img_aug[k] = img_aug[k].expand([3, 224, 224])
                    img_aug[k] = img_aug[k].unsqueeze(0)

                img_aug = torch.cat(img_aug)

            sample = [img_aug, label]
            return sample

        # aift trainning stage
        elif (self.phase == 'aift') and (self.show_full_label == False):
            image = transforms.Scale((70,70))(image)
            image = RandCrop()(image)
            image = RandRotate()(image)

        if self.transform:
            image = self.transform(image)

        image = image.expand([3, 224, 224])
        sample = [image, label]

        return sample

    def add_labeled_candidate(self, candidates_need_label):
        for i in range(len(candidates_need_label)):

            # self.aift_labeled contains the index of candidates that are labeled
            if not (self.num_to_index[candidates_need_label[i]] in self.aift_labeled):
                self.aift_labeled.append(self.num_to_index[candidates_need_label[i]])

    def has_index_from_num(self, i):
        return self.num_to_index[i] in self.aift_labeled

if __name__ == '__main__':
    # candidates = pd.read_csv('/home/hyacinth/project/luna16/candidates.csv')
    trans = transforms.Compose([
        transforms.Scale((224, 224)),
        ToTensor()
    ])

    dataset = Luna2DData('train', trans)

    for i in range(len(dataset)):
        dataset.__getitem__(i,showpic=True)
    # data = RandRotate()(dataset[1])
