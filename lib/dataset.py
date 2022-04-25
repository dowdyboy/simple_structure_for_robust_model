import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import random

from dowdyboy_lib.rand import wheel_rand_index, wheel_ratio


class AdvMixDataset(Dataset):

    def __init__(self, clean_dir, adv_dir_list, adv_ratio=0.5, adv_sample_weights=None, is_train=True, transforms=None, ):
        super(AdvMixDataset, self).__init__()
        self.transforms = transforms
        self.clean_dataset = ImageFolder(clean_dir, transform=transforms,)
        self.adv_dataset_list = [ImageFolder(adv_dir, transform=transforms, ) for adv_dir in adv_dir_list]
        self.adv_ratio = adv_ratio
        self.adv_sample_weights = [1. for _ in range(len(adv_dir_list))] if adv_sample_weights is None else adv_sample_weights
        self.is_train = is_train

    def __getitem__(self, idx):
        if self.is_train:
            if random.random() < self.adv_ratio:
                adv_idx = wheel_rand_index(self.adv_sample_weights)
                return self.adv_dataset_list[adv_idx][idx] + (1, )
            else:
                return self.clean_dataset[idx] + (0, )
        else:
            clean_thres = int((1. - self.adv_ratio) * len(self.clean_dataset))
            adv_thres = len(self.clean_dataset) - clean_thres
            if idx < clean_thres:
                return self.clean_dataset[idx] + (0, )
            else:
                adv_ratio = wheel_ratio(self.adv_sample_weights)
                for adv_idx, r in enumerate(adv_ratio):
                    if idx < clean_thres + int(adv_thres * r):
                        return self.adv_dataset_list[adv_idx][idx] + (1, )
        return self.clean_dataset[idx] + (0, )

    def __len__(self):
        return len(self.clean_dataset)

