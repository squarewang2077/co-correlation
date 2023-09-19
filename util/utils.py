import os 
import pandas as pd 
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
from torch.utils.data import random_split

class InputResize(nn.Module):
    def __init__(self, alpha, beta, device):
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
        self.beta = torch.tensor(beta, dtype=torch.float32).to(device)

    def forward(self, x):
        alpha = self.alpha[None,:, None, None].expand_as(x)
        beta = self.beta[None,:, None, None].expand_as(x)
        x = (x - alpha)/beta
        return x

class InputMinus(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):

        return self.module(x) - x


class Logger:
    def __init__(self):
        root = './log'
        self.logging_roots = {}
        self.logging_roots['ckp'] = f'{root}/' # ckp is the folder to save trained networks 
        self.logging_roots['results'] = f'{root}/'  # training_results is the folder for training results
        self.logging_roots['adv_examples'] = f'{root}/examples/' # adv_exampels is a folder to save the imgs 

        # initialized when _load_from_log is called 
        self.logging_files = {} # to store the loaded files

        # initalized when _load_ckp is called 
        # it will be saved when _save_ckp is called
        self.ckp_state_dict = {} # to store the information from ckp 

    def _rander_path(self, file_path):
        # notice that file_path contains the folder name, e.g., results/.../....
        folder = file_path.split('/')[0]
        file_path = self.logging_roots[folder] + file_path
        file_dir = '/'.join(file_path.split('/')[:-1])        
        return file_dir, file_path

    def _save_to_log(self, data_dict, index_name, file_path):
        file_dir, file_path = self._rander_path(file_path)
        if isinstance(index_name, str):
            file = pd.DataFrame(data_dict).set_index(index_name)
        else: 
            file = pd.DataFrame(data_dict, index=None)

        if not os.path.exists(file_path):
            Path(file_dir).mkdir(parents=True, exist_ok=True)                        
            file.to_csv(file_path, mode='a', header=True)
        else: 
            file.to_csv(file_path, mode='a', header=False)

    def _load_from_log(self, file_path, **kwargs):
        _, file_path = self._rander_path(file_path)
        file_name = file_path.split('/')[-1][:-4]
        if os.path.exists(file_path):
            self.logging_files[file_name] = pd.read_csv(file_path, **kwargs) # the loaded csv data are stored in self.logging_files

    def _save_ckp(self, ckp_path):
        # to save the ckp not only the trained network 
        ckp_dir, ckp_path = self._rander_path(ckp_path)

        if not os.path.exists(ckp_path):
            Path(ckp_dir).mkdir(parents=True, exist_ok=True)                        
            torch.save(self.ckp_state_dict, ckp_path)
        else:
            torch.save(self.ckp_state_dict, ckp_path)

    def _load_ckp(self, ckp_path):
        _, ckp_path = self._rander_path(ckp_path)
        assert os.path.exists(ckp_path), "checkpoint not found"
        self.ckp_state_dict = torch.load(ckp_path) # load the trained model 

    def _save_imgs(self, img, img_path):

        img_dir, img_path = self._rander_path(img_path)
        if not os.path.exists(img_dir):
            Path(img_dir).mkdir(parents=True, exist_ok=True)
            save_image(img, img_path)
        else: 
            save_image(img, img_path)                    


def subsample(dataset, frac_size, batch_size, random_seed=99):
    # take a subset of the dataset to analyze 
    _, sub_dataset = random_split(dataset, [len(dataset) - frac_size, frac_size], \
                                  generator=torch.Generator().manual_seed(random_seed))
    sub_dataloader = DataLoader(sub_dataset,
                            batch_size=batch_size, # each batch contains only one image 
                            pin_memory=True,
                            num_workers=4,
                            shuffle=False)
    return sub_dataloader

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(logits, labels, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk) # e.g., topk = (1, 5), maxk = 5
    batch_size = labels.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True) # output is value, index
    correct = pred.eq(labels.reshape(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:,:k].sum()
        res.append(correct_k / batch_size)
    return res

def dist_of(imgs, adv_imgs, Lp_norm):
    '''
    Args: 
    delta_imgs of size (B C H W)
    Lp_norm is the norm used
    '''
    delta_imgs = imgs.detach() - adv_imgs.detach()
    delta_imgs = delta_imgs.flatten(start_dim=1).abs()
    if Lp_norm == 'Linf':
        dists = delta_imgs.max(dim=1)[0]
    if Lp_norm == 'L2':
        dists = (delta_imgs**2).sum(dim=1).sqrt()
    if Lp_norm == 'scaled_L2':
        dists = ((delta_imgs**2)/delta_imgs.shape[1]).sum(dim=1).sqrt()    

    return dists

class Measure(object):
    def __init__(self, Lp_norms, topk = (1,)):

        self.Lp_norms= Lp_norms
        self.topk = topk
        self.losses = AverageMeter()
        self.acc_dict = {}
        self.dist_info_dict = {}
        for k in topk:
            self.acc_dict[f'acc_top{k}'] = AverageMeter()
        for norm in Lp_norms:
            self.dist_info_dict[f'avg_{norm}'] = AverageMeter()
            self.dist_info_dict[f'avg_square_{norm}'] = AverageMeter()

    def update_info(self, logits, labels, loss, \
                    imgs, adv_imgs):

        self.losses.update(loss, len(imgs))
        acc_list = accuracy(logits, labels, self.topk)
        for i, k in enumerate(self.topk):
            self.acc_dict[f'acc_top{k}'].update(acc_list[i], len(imgs))

        for norm in self.Lp_norms:
            self.dist_info_dict[f'avg_{norm}'].update(dist_of(imgs, adv_imgs, norm).mean(), len(imgs))
            self.dist_info_dict[f'avg_square_{norm}'].update((dist_of(imgs, adv_imgs, norm)**2).mean(), len(imgs))
            

      

