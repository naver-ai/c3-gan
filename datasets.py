import os
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms



class Dataset(data.Dataset):
    def __init__(self, data_dir, mode='train'):

        self.transform = transforms.Compose([transforms.Resize(128),
                                             transforms.RandomCrop(128),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_aug = transforms.Compose([transforms.RandomResizedCrop(128, scale=(0.2, 1.0)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                                 transforms.RandomGrayscale(0.2),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.data_dir = data_dir
        self.img_paths, self.labels = self.load_filenames(data_dir, mode)

        if mode=='train':
            self.iterator = self.prepare_training_pairs
        else:
            self.iterator = self.prepare_test_pairs

    def load_filenames(self, data_dir, mode):
        if mode == 'train':
            with open(os.path.join(data_dir, 'trainset.txt'), 'r') as f:
                data = f.readlines()
        else:
            with open(os.path.join(data_dir, 'testset.txt'), 'r') as f:
                data = f.readlines()
        img_paths = [os.path.join(data_dir, 'images', _.split()[0]) for _ in data]
        labels = [int(_.split()[-1]) for _ in data]
        return img_paths, labels
    
    def prepare_training_pairs(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img_ = self.transform(img)
        img_aug_ = self.transform_aug(img)
        return img_, img_aug_

    def prepare_test_pairs(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = transforms.Resize(128)(img)
        img = transforms.CenterCrop(128)(img)
        img = self.norm(img)
        return img, self.labels[index]

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.img_paths)
