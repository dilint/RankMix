import os.path as osp
import copy
from stringprep import in_table_d2
import sys, os, glob
import numbers
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as tr
import torchvision.transforms.functional as VF
from sklearn.utils import shuffle

class ClassDataset(Dataset):
    def __init__(self, data, args, transforms=None, test=False):
        self.data = data
        self.args = args            
        self.transforms = transforms

    def __getitem__(self, index):              
        csv_file_df = self.data.iloc[index]
        if self.args.dataset == 'TCGA-lung-default':
            feats_path = 'datasets/tcga-dataset/tcga_lung_data_feats-npy/' + csv_file_df.iloc[0].split('/')[1] + '.npy'
        else:            
            feats_path = os.path.join(
                os.path.dirname(csv_file_df.iloc[0]) + '-npy', 
                os.path.basename(csv_file_df.iloc[0]).replace('.csv', '.npy')
            )
        y = int(csv_file_df.iloc[1])
        
        feats = np.load(feats_path)            
        label = np.zeros(self.args.num_classes)
        
        if self.args.num_classes==1:
            label[0] = y
        else:
            if y<=(len(label)-1):
                label[y] = 1
        if self.transforms is not None:
            img = self.transforms(img)
        
        return torch.Tensor(label), torch.Tensor(feats)

    def __len__(self):
        return len(self.data)



class ClassDatasetRemix(Dataset):
    def __init__(self, feats, labels, args, mode='train', transforms=None, test=False):        
        self.feats = feats
        self.labels = labels
        self.args = args
        self.mode = mode            
        self.transforms = transforms

        
        if self.mode == 'train':
            self.feats = torch.Tensor(self.feats)
            self.labels = torch.LongTensor(self.labels)
        

    def __getitem__(self, index):
        # load image and labels 
        if self.mode == 'train':
            feat = self.feats[index]
        else:            
            feat = np.load(self.feats[index].split(',')[0])
          
        label = np.zeros(self.args.num_classes)
        if self.args.num_classes==1:
            label[0] = self.labels[index]
        else:
            if int(self.labels[index])<=(len(label)-1):
                label[int(self.labels[index])] = 1
        if self.transforms is not None:
            img = self.transforms(img)        
        return torch.Tensor(label), torch.Tensor(feat)

    def __len__(self):
        return len(self.feats)

# https://github.com/huanghoujing/pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
class ComboIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [next(loader_iter) for loader_iter in self.loader_iters]
        # batches = [loader_iter.next() for loader_iter in self.loader_iters] # different torch version
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)

class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches

def get_sampling_probabilities(class_count, mode='instance', ep=None, n_eps=None):
    '''
    Note that for progressive sampling I use n_eps-1, which I find more intuitive.
    If you are training for 10 epochs, you pass n_eps=10 to this function. Then, inside
    the training loop you would have sth like 'for ep in range(n_eps)', so ep=0,...,9,
    and all fits together.
    '''
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5 # 1/2
    elif mode == 'cbrt':
        q = 0.125 # 1/8
    elif mode == 'prog':
        assert ep != None and n_eps != None, 'progressive sampling requires to pass values for ep and n_eps'
        relative_freq_imbal = class_count ** 0 / (class_count ** 0).sum()
        relative_freq_bal = class_count ** 1 / (class_count ** 1).sum()
        sampling_probabilities_imbal = relative_freq_imbal ** (-1)
        sampling_probabilities_bal = relative_freq_bal ** (-1)
        return (1 - ep / (n_eps - 1)) * sampling_probabilities_imbal + (ep / (n_eps - 1)) * sampling_probabilities_bal
    else: sys.exit('not a valid mode')

    relative_freq = class_count ** q / (class_count ** q).sum()
    sampling_probabilities = relative_freq ** (-1)

    return sampling_probabilities

def modify_loader(loader, mode, dataset_type=None, ep=None, n_eps=None):    
    label_col = loader.dataset.data.iloc[:,1]
    class_count = np.unique(label_col, return_counts=True)[1]
    sampling_probs = get_sampling_probabilities(class_count, mode=mode, ep=ep, n_eps=n_eps)
    sample_weights = sampling_probs[label_col]

    mod_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    mod_loader = DataLoader(loader.dataset, batch_size = loader.batch_size, sampler=mod_sampler, num_workers=loader.num_workers)
    return mod_loader

def get_combo_loader(loader, base_sampling='instance', dataset_type=None):
    if base_sampling == 'instance':
        imbalanced_loader = loader
    else:
        imbalanced_loader = modify_loader(loader, mode=base_sampling, dataset_type=dataset_type)

    balanced_loader = modify_loader(loader, mode='class', dataset_type=dataset_type)
    combo_loader = ComboLoader([imbalanced_loader, balanced_loader])
    return combo_loader


class C16Dataset(Dataset):

    def __init__(self, file_name, file_label, root, num_classes=9, persistence=False,keep_same_psize=0):
        """
        参数
            file_name: WSI pt文件名列表
            file_label: WSI标签列表
            root: WSI pt文件根目录
            persistence: 是否将所有pt文件在init()中加载到内存中
            keep_same_psize: 是否保持每个样本的patch数量一致
            is_train: 是否为训练集
        """
        super(C16Dataset, self).__init__()
        self.file_name = file_name
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.num_classes = num_classes
        self.data = pd.DataFrame({
            'features': list(self.file_name),  # 将每行特征保存为列表元素
            'label': self.slide_label
        })
        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]

    def __len__(self):
        return self.size
    
    def _pading_tensor(self, tensor):
        """
        为[N,M,C]或者[N,C]的tensor进行padding，其中N为不定长，M，C为定长，使其变为[self.keep_same_psize,M,C]
        参数:
            tensor: [N,M,C]的tensor
        返回:
            padded_tensor: [self.keep_same_psize,M,C]的tensor
            mask: [self.keep_same_psize, M]的mask，值为0或1，表示是否为有效数据
        """
        keep_same_psize = self.keep_same_psize
        # 获取当前tensor的长度
        N = tensor.shape[0]
        # 创建一个1000x25x256的0矩阵
        padded_tensor = torch.zeros((keep_same_psize, *(tensor.shape[1:]))) 
        # 创建一个长度为1000的binary mask
        mask = torch.zeros(keep_same_psize)
        # 如果N小于1000，填充数据并设置mask
        if N < keep_same_psize:
            padded_tensor[:N] = tensor
            mask[:N] = 1
        else:  # 如果N大于1000，截断数据并设置mask
            padded_tensor = tensor[:keep_same_psize]
            mask = torch.ones(keep_same_psize)
        if len(tensor.shape) > 2:
            mask = mask.unsqueeze(-1).expand(-1, padded_tensor.shape[1])
        return padded_tensor, mask
    
    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            if "pt" in os.listdir(self.root):
                dir_path = os.path.join(self.root,"pt")
            else:
                dir_path = self.root
            file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
            features = torch.load(file_path, map_location='cpu', weights_only=False)
        mask = torch.ones(len(features))
        if self.keep_same_psize > 0:
            features, mask = self._pading_tensor(features)
        label = int(self.slide_label[idx])
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return label.float(), features