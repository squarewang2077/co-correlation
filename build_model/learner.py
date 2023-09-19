import time
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from util import *
from build_model.util import *

####
from analyze_model import *
from analyze_model.util import *
from ete3 import Tree
####


class ShadowTree(Tree):
    def __init__(self, newick=None, format=0, dist=None, support=None, name=None, quoted_node_names=False):
        super().__init__(newick, format, dist, support, name, quoted_node_names)
        self.dist4child = None 

    def assign_child_params(self):        
        self.dist4child = self.dist + 1

    def add_child(self, child=None, name=None, dist=None, support=None):
        dist = self.dist4child
        return super().add_child(child, name, dist, support)



class Learner(Logger):
    def __init__(self, named_dataset, named_network,\
                       optimizer, lr_scheduler, loss_fn, device, \
                       net_type, ckp_folder = 'ckp/', result_path='results/training_results.csv'):
        super(Learner, self).__init__()
        
        self.dataset_name, self.train_set, self.test_set = named_dataset 
        self.net_name, self.net = named_network
        # self.attack, self.attack_name = named_attack
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device

        self.net_type = net_type
        self.ckp_folder = ckp_folder
        self.result_path = result_path

        # adding value during training
        # notice that the ckp_state_dict is initialized in Logger 
        self.training_results = {
            'dataset': [],
            'net_name': [],
            'epoch': [],
            'train_loss': [],
            'train_acc': [], 
            'val_loss': [], 
            'val_acc': []
        }

    def epoch_run(self, dataloader, backward_pass=False):
        '''
        This function is to do each epoch run of the dataset
        '''
        total_loss = 0
        total_corrects = 0
        
        num_of_batches = 0
        num_of_datas = 0
        for imgs, labels in tqdm(dataloader):
            num_of_batches += 1 
            num_of_datas += len(imgs) # more accurately collect the numbers of the data

            imgs, labels = imgs.to(self.device), labels.to(self.device)

            logits = self.net(imgs) # in logit space 
            loss = self.loss_fn(logits, labels)
            if backward_pass:

                # check whether use sam to optimize the network
                is_sam = self.optimizer.__class__.__name__ == 'SAM' 
                if is_sam:
                    def closure():
                        loss = self.loss_fn(self.net(imgs), labels)
                        loss.backward()
                        return loss

                self.optimizer.zero_grad()
                loss.backward()
                if is_sam:
                    self.optimizer.step(closure)
                else: 
                    self.optimizer.step()

                # check whether using learning scheduler 
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
 
            preds = logits.argmax(dim=1) # the prediction 
            total_loss += loss.detach().item() # total loss 
            total_corrects += preds.eq(labels).sum().detach().item() # total corrects 

        # average loss and acc
        loss = total_loss/num_of_batches
        acc = total_corrects/num_of_datas

        return loss, acc

    def save_ckp(self, epoch):
        self.ckp_state_dict['net_state_dict'] = self.net.state_dict()
        self.ckp_state_dict['opt_state_dict'] = self.optimizer.state_dict()
        self._save_ckp(f'{self.ckp_folder}{self.net_name}({epoch})_on_{self.dataset_name}.pth')
 
    def load_ckp(self, epoch):
        self._load_ckp(f'{self.ckp_folder}{self.net_name}({epoch})_on_{self.dataset_name}.pth')
    
    def save_to_log(self):
        self._save_to_log(self.training_results, 'epoch', self.result_path)                    

    def train(self, start_epoch: int, end_epoch: int, batch_size: int, \
              eval_freq: int = None, ckp_epochs: list = [], verbose: bool=True, \
              analyze_batch_size:int=20, decomp_contrs:dict=None, fraction_size:int=1024, analyze_freq:int=1, \
              eval_method:str='msv_by_power_iteration'):                
        # load the train_set 
        train_loader = DataLoader(self.train_set,
                                  batch_size=batch_size,
                                  num_workers=4,
                                  pin_memory=True)
        # load the test_set                                
        test_loader = DataLoader(self.test_set,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True,
                                shuffle=False) 

        assert start_epoch >= 1, "start epoch has to start from 1 instead of 0 or below"

        # load and continue training for the perviously trained model
        if start_epoch > 1:
            self.load_ckp(start_epoch)              
            self.net.load_state_dict(self.ckp_state_dict['net_state_dict'])
            self.optimizer.load_state_dict(self.ckp_state_dict['opt_state_dict'])

        self.net.train() # training mode
        for epoch in range(start_epoch, end_epoch+1): # the training will include start and end epoch 
            
            # train the net  
            start_time = time.time()
            train_loss, train_acc = self.epoch_run(train_loader, backward_pass=True)        
            end_time = time.time()
            if verbose: # showing the detail of the training 
                print(f'time: {(end_time - start_time):.3f} train: epoch {epoch} loss {train_loss:.3f}, acc {train_acc:.3f}')

            # save the net 
            if epoch in ckp_epochs:
                self.save_ckp(epoch)

            # save the training results     
            if eval_freq is not None:
                if epoch % eval_freq == 0: # show the training results for given epochs                    
                    val_loss, val_acc = self.evaluate(test_loader) # evaluate the testset
                    if verbose: # showing the detailed test results 
                        print(f'val: epoch {epoch} loss {val_loss} acc {val_acc}')

                    self.training_results['dataset'].append(self.dataset_name)
                    self.training_results['net_name'].append(self.net_name)
                    self.training_results['epoch'].append(epoch)
                    self.training_results['train_loss'].append(train_loss)
                    self.training_results['train_acc'].append(train_acc)
                    self.training_results['val_loss'].append(val_loss)
                    self.training_results['val_acc'].append(val_acc)
            
            if epoch % analyze_freq == 0:
                for depth, control in decomp_contrs.items():
                    control = control.copy() 
                    self.analyze(analyze_batch_size, fraction_size, eval_method, depth, control, epoch)

        # save the self.training_results
        self.save_to_log()                    
                                            
    def analyze(self, batch_size, fraction_size, eval_method, analyze_depth, decompose_control, epoch):

        analysis = BaseAnalyser(named_dataset=(self.dataset_name, self.test_set), named_network=(self.net_name, self.net), \
                                named_attack=(None, None), analyze_depth=analyze_depth, device=DEVICE, result_folder=f'results/{self.net_type}/epoch-{epoch}')

        self.show_architecture()
        analysis.decompose(decompose_control)
        analysis.subsample(fraction_size, batch_size) 
        ana_result = analysis.analyze(False, eval_method, iters=30)

        # ana_stat_pd = (pd.DataFrame(ana_result).iloc[:,:-2]**2).mean().to_frame().T
        # ana_stat_pd['epoch'] = int(epoch)
        # ana_stat_pd['acc'] = (torch.tensor(ana_result['label']) == torch.tensor(ana_result['prediction'])).sum().item()/fraction_size
        # ana_stat_pd.set_index('epoch', inplace=True)

        analysis.save_to_log(ana_result, eval_method, iters=30) 


    def show_architecture(self):
        shadowtree = ShadowTree(name='root', dist=2)
        nntree = nnTree({'named_module': (self.net_name, self.net), 'tree4plot': shadowtree})
        print(nntree.dict_node['tree4plot'].get_ascii(attributes=['name'], show_internal=True))


    def evaluate(self, dataloader):
                
        self.net.eval() # evaluation model
        with torch.no_grad():
            loss, acc  = self.epoch_run(dataloader)
        return loss, acc


