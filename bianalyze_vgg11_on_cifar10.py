import math
import argparse 
import torch
import torch.nn as nn 
import torch.nn.functional as F

from util import * 
from build_model import *
from build_model.util import *
from analyze_model import *
from analyze_model.util import *

from ete3 import Tree
import timm 

class ShadowTree(Tree):
    def __init__(self, newick=None, format=0, dist=None, support=None, name=None, quoted_node_names=False):
        super().__init__(newick, format, dist, support, name, quoted_node_names)
        self.dist4child = None 

    def assign_child_params(self):        
        self.dist4child = self.dist + 1

    def add_child(self, child=None, name=None, dist=None, support=None):
        dist = self.dist4child
        return super().add_child(child, name, dist, support)

def main():

    parser = argparse.ArgumentParser(description='train_attack_analysis_MLPs')

    # the class of the model, e.g., vit or covit  
    parser.add_argument('--net_type', type=str, default='vgg11', 
                        help='the type of the neural network, e.g., vit or covit')

    # dataset configuration 
    parser.add_argument('--dataset', type=str, default="cifar10", 
                        help='the data set to be trained with')
    parser.add_argument('--transform_resize', type=int, default=224, 
                    help='transform the inputs: resize the resolution')

    # network setting 
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='pretraining or not',)       
    parser.add_argument('--n_cls', type=int, default=10, 
                        help='the number of class')
    parser.add_argument('--init_par', type=float, default=-0.2, 
                        help='the paramters that contrain the scale of weight initialization')
    
    # setting for training process 
    ## basic setting 
    parser.add_argument('--start_epoch', type=int, default=1, 
                        help='the start_epoch for training')
    parser.add_argument('--end_epoch', type=int, default=50, 
                        help='the end epoch')
    parser.add_argument('--batch_size', type=int, default=1024, 
                        help='the batch size for training and validating')
    ## checkpoint setting
    parser.add_argument('--eval_freq', type=int, default=2, 
                        help='the frequence to evaluate the network')
    parser.add_argument('--ckp_epochs', nargs='+', type=int, default=[50], 
                        help='the list to store the network')                        
    parser.add_argument('--analyze_freq', type=int, default=2, 
                        help='the frequence to evaluate the network')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='show the training and evaluation details',)       

    # arguments for risk evluation 
    parser.add_argument('--analyze_batch_size', type=int, default=100, 
                        help='the batch size for analyzing the model')
    parser.add_argument('--fraction_size', type=int, default=1000, 
                        help='the subset of the test data to be attacked')
    parser.add_argument('--eval_method', type=str, default='msv_by_power_iteration', 
                        help='the data set to be trained with')

    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the train loader, prepare for training 
    train_set, test_set = get_dataset(args, norm=True)
    net_name, net = get_network(args)
    net = net.to(DEVICE)    

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, net.parameters()), lr=0.0005)
    lr_scheduler = None

    decom_ctls = {0:[[0]], 1:[[1]]}
    # initialization of the learner which is used to train and validate the network 
    learner = Learner((args.dataset, train_set, test_set), (net_name, net), optimizer, lr_scheduler, nn.CrossEntropyLoss(), device=DEVICE, \
                       net_type=args.net_type, ckp_folder = f'ckp/{args.net_type}/', result_path=f'results/{args.net_type}/training_results.csv')
    learner.train(args.start_epoch, args.end_epoch, args.batch_size, args.eval_freq, args.ckp_epochs, args.verbose,\
                  args.analyze_batch_size, decom_ctls, args.fraction_size, args.analyze_freq, args.eval_method)
 
if __name__ == '__main__':
    main()