# path = 'IMagenet/tiny-imagenet-200/'
import time
# import scipy.ndimage as nd
from skimage import io
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensor
from albumentations import Compose, RandomCrop,PadIfNeeded,Flip, Normalize, HorizontalFlip, Cutout, VerticalFlip, Rotate, RGBShift
from albumentations.pytorch import ToTensor

path = 'IMagenet/tiny-imagenet-200/'

def get_id_dictionary():
    id_dict = {}
    for i, line in enumerate(open( path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict
  
def get_class_to_id_dict():
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open( path + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])      
    return result

def get_data(id_dict):
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    for key, value in id_dict.items():
        train_data += [io.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), as_gray=False, pilmode='RGB') for i in range(500)]
        # train_labels_ = np.array([[0]*200]*500)
        # train_labels_[:, value] = 1
        # train_labels += train_labels_.tolist()
        train_labels+= 500 * [value]

    for line in open( path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        train_data.append(io.imread( path + 'val/images/{}'.format(img_name) ,as_gray=False, pilmode='RGB'))
        # test_labels_ = np.array([[0]*200])
        # test_labels_[0, id_dict[class_id]] = 1
        # train_labels += test_labels_.tolist()
        train_labels.append(id_dict[class_id])

    print('finished loading data, in {} seconds'.format(time.time() - t))
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

class albumCompose:
    def __init__(self):
        self.albumentations_transform = Compose({
            # PadIfNeeded(min_height=40,min_width=40),
            # RandomCrop(32,32),
            Flip(),
            VerticalFlip(p=0.5),
            Cutout(max_h_size=8,max_w_size=8,num_holes=1),
            Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        })
    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return torch.tensor(img, dtype=torch.float)

class albumCompose_test:
    def __init__(self):
        self.albumentations_transform = Compose({
        Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        })
    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return torch.tensor(img, dtype=torch.float)


class TinyImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i]
        data = self.transforms(data)
        return (data, self.y[i])

def getDataset():
    SEED = 1
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    # For reproducibility
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    train_data, train_labels, test_data, test_labels = get_data(get_id_dictionary())
    c = list(zip(train_data, train_labels))
    random.shuffle(c)
    train_data, train_labels = zip(*c)
    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)
    
    trainlen = int(len(train_data)*0.7)
    print((trainlen))
    trd = train_data[:trainlen]
    tsd = train_data[trainlen:]
    trl =train_labels[:trainlen]
    tsl =train_labels[trainlen:]
    print(len(tsl))
    train_data = TinyImageDataset(trd, trl, albumCompose())
    test_data = TinyImageDataset(tsd, tsl, albumCompose_test())

    # dataloaders
    trainloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2)
    testloader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=2)
    return trainloader, testloader, device
