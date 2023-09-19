import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import math
from tqdm import tqdm

from attack_model.util import *
from util import * 

class Attack_Ensembles(Logger):
    def __init__(self, named_dataset, named_network, named_attack, Lp_for_dist, topk, num_saving, loss_fn, device):
        super(Attack_Ensembles, self).__init__()

        self.dataset_name, self.dataset = named_dataset
        self.net_name, self.net = named_network
        self.atk_name, self.atk = named_attack
        self.Lp_for_dist = Lp_for_dist
        self.loss_fn = loss_fn
        self.device = device
        self.topk = topk

        self.num_saving = num_saving # the number of saving data
        
        # initalized when attack 
        self.num_saving_per_batch = None 
        self.batch_index = None

        # initialized when subsample is called 
        self.dataloader = None 

    def subsample(self, frac_size, batch_size):
        self.dataloader = subsample(self.dataset, frac_size, batch_size)

    def save_imgs(self, adv_imgs, ground_truth, preds):
        if self.num_saving > 0:
            index = torch.randint(0, len(adv_imgs), (self.num_saving_per_batch,)) # randomly choose numbers 
            adv_preds = self.net(adv_imgs).argmax(dim=1)
            for i in range(self.num_saving_per_batch):
                img_index = self.batch_index * self.num_saving_per_batch + i
                img_name = f'{self.dataset_name}({img_index})_{ground_truth[index[i]]}->{preds[index[i]]}->{adv_preds[index[i]]}.png'

                self._save_imgs(adv_imgs[index[i]], f'adv_examples/{self.net_name}/{self.atk_name}/{self.dataset_name}/{img_name}')
            self.num_saving -= self.num_saving_per_batch
        else: 
            pass 


    def attack(self):
        self.batch_index = 0
        self.num_saving_per_batch = math.ceil(self.num_saving/len(self.dataloader))

        # tot_loss = 0
        # tot_corrects = 0

        # num_of_batches = 0
        # num_of_datas = 0

        # # for average and std for img and adv_img distance
        # dist_dict = dict()
        # for Lp_norm in self.Lp_for_dist:         
        #     dist_dict[Lp_norm] = {'mean': [], 'std':[]}
        # dist_dict['cor_term'] = []
        
        # initializtion for the robust acc. and loss 
        result = {}
        result['attacks'] = [self.atk_name]
        result['networks'] = [self.net_name.split('_on_')[0]]
        result['dataset'] = [self.dataset_name]

        # measure #
        measure = Measure(Lp_norms=self.Lp_for_dist, topk=self.topk)
        # measure #

        for imgs, labels in tqdm(self.dataloader):

            imgs, labels = imgs.to(self.device), labels.to(self.device)

            # imgs = imgs.to(self.device)

            # tot_corrects_check = self.net(imgs).argmax(dim=1).eq(labels).sum().detach().item() # for debug

            adv_imgs = self.atk(imgs, labels).to(self.device) # implement the attack 

            # for robust loss  
            adv_logits = self.net(adv_imgs)
            adv_loss = self.loss_fn(adv_logits, labels).detach().item()
            # tot_loss += adv_loss # total loss  

            # for robust acc 
            # tot_corrects += adv_logits.argmax(dim=1).eq(labels).sum().detach().item() # total corrects 

            # measure #
            measure.update_info(adv_logits.detach(), labels.detach(), adv_loss, imgs.detach(), adv_imgs.detach())
            # measure # 
            
            # # for the distance
            # for Lp_norm in self.Lp_for_dist: 
            #     dists = dist_of(imgs, adv_imgs, Lp_norm)
            #     dist_dict[Lp_norm]['mean'].append(torch.tensor(dists).mean().item())
            #     dist_dict[Lp_norm]['std'].append(torch.tensor(dists).std().item())
            # dist_dict['cor_term'].append(len(adv_imgs)/len(self.dataloader.dataset))

            # save a fraction of perturbed images  
            self.save_imgs(adv_imgs, labels, self.net(imgs).argmax(dim=1))
            self.batch_index += 1 # this is for naming the saved image 

            # num_of_batches += 1 
            # num_of_datas += len(adv_imgs) # more accurately collect the numbers of the data

        # # average loss and acc
        # result['rob_loss'] = [tot_loss/num_of_batches]
        # result['rob_acc'] = [tot_corrects/num_of_datas]
        # for Lp_norm in self.Lp_for_dist:
        #     result[Lp_norm + '_dist_mean'] = [(torch.tensor(dist_dict['cor_term'])@torch.tensor(dist_dict[Lp_norm]['mean'])).item()]
        #     result[Lp_norm + '_dist_std'] = [(torch.tensor(dist_dict['cor_term'])@torch.tensor(dist_dict[Lp_norm]['std'])).item()] 

        # measure #
        result['rob_loss'] = [measure.losses.avg]
        for k in self.topk:
            result[f'rob_acc_top{k}'] = [measure.acc_dict[f'acc_top{k}'].avg.item()]
        for norm in self.Lp_for_dist:
            result[norm + '_dist_mean'] = [measure.dist_info_dict[f'avg_{norm}'].avg.item()]
            result[norm + '_dist_std'] = [(measure.dist_info_dict[f'avg_square_{norm}'].avg - measure.dist_info_dict[f'avg_{norm}'].avg**2).sqrt().item()] 
        # measure #


        self._save_to_log(result, 'attacks', f'results/attacking_results.csv')

        
 

