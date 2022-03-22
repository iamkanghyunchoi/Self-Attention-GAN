import torch
import torchvision.datasets as dsets
from torchvision import transforms

from torch.utils.data import Dataset

from glob import glob
from typing import List, Union, Tuple
import os
from PIL import Image

import copy
from torchvision.transforms.functional import to_tensor

class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        return dataset


    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2,
                                              drop_last=True)
        return loader


class GeneratorDataset(Dataset):
    def __init__(self, G, z_dim, batch_size=1):
        self.G = G
        self.z_dim = z_dim
        self.batch_size=batch_size
    
    def __len__(self):
        return 50000 #//self.batch_size
    
    def __getitem__(self, index):
        # print(index)
        return self.G(torch.randn(self.batch_size, self.z_dim).cuda())[0] #####

def squeeze_batch(samples):
    inputs = torch.stack(samples,dim=0).squeeze(0)
    return inputs

# def omit_label_batch(samples):
#     print(samples)
#     only_inputs = [sample[0] for sample in samples]
#     print(only_inputs)
#     inputs = torch.stack(only_inputs,dim=0)
#     return inputs

class ImageDataset(Dataset):
    def __init__(self, path, exts=['png', 'jpg'], transform=None):
        self.transform=transform
        self.paths = []
        for ext in exts:
            self.paths.extend(
                list(glob(os.path.join(path, '*.%s' % ext))))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        if self.transform is None:
            tensor = copy.deepcopy(to_tensor(image))
        else:
            tensor = copy.deepcopy(self.transform(image))
        image.close()
        return tensor