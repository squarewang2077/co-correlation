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
    parser.add_argument('--net_type', type=str, default='twolayer', choices=['twolayer'], 
                        help='the type of the neural network, e.g., vit or covit')

    # configuration for shallow model
    parser.add_argument('--width_1', type=int, default=16, 
                        help='the width of layer 1')
    parser.add_argument('--width_2', type=int, default=16, 
                        help='the width of layer 2')
    parser.add_argument('--act_name', type=str, default='relu', 
                        help='the activation')
    parser.add_argument('--img_size', nargs='+', type=int, default=[1, 28, 28], 
                        help='the input image size, e.g., (224, 224), including channles')
    parser.add_argument('--n_classes', type=int, default=10, 
                        help='the number of classes')

    # dataset configuration 
    parser.add_argument('--dataset', type=str, default="minst", 
                        help='the data set to be trained with')
    parser.add_argument('--transform_resize', type=int, default=224, 
                    help='transform the inputs: resize the resolution')


    # setting for training process 
    ## basic setting 
    parser.add_argument('--start_epoch', type=int, default=1, 
                        help='the start_epoch for training')
    parser.add_argument('--end_epoch', type=int, default=50, 
                        help='the end epoch')
    parser.add_argument('--batch_size', type=int, default=512, 
                        help='the batch size for training and validating')
    ## checkpoint setting
    parser.add_argument('--eval_freq', type=int, default=1, 
                        help='the frequence to evaluate the network')
    parser.add_argument('--ckp_epochs', nargs='+', type=int, default=[50], 
                        help='the list to store the network')                        
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='show the training and evaluation details',)       

    # arguments for risk evluation 
    parser.add_argument('--analyze_batch_size', type=int, default=200, 
                        help='the batch size for analyzing the model')
    parser.add_argument('--fraction_size', type=int, default=10000, 
                        help='the subset of the test data to be attacked')
    parser.add_argument('--eval_method', type=str, default='msv_by_power_iteration', 
                        help='the data set to be trained with')

    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the train loader, prepare for training 
    train_set, test_set = get_dataset(args, norm=True)
    net_name, net = get_network(args)
    net = net.to(DEVICE)    

    # set up attacking 
    # attack_name, attack = get_attack(net, args)

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, net.parameters()), lr=0.003)
    lr_scheduler = None

    decom_contrs = {0:[[0]], 1:[[1]]}
    # initialization of the learner which is used to train and validate the network 
    learner = Learner((args.dataset, train_set, test_set), (net_name, net), optimizer, lr_scheduler, nn.CrossEntropyLoss(), device=DEVICE, \
                       net_type=args.net_type, ckp_folder = f'ckp/{args.net_type}/', result_path=f'results/{args.net_type}/training_results.csv')
    learner.train(args.start_epoch, args.end_epoch, args.batch_size, args.eval_freq, args.ckp_epochs, args.verbose,\
                  args.analyze_batch_size, decom_contrs, args.fraction_size, args.eval_method)
 

if __name__ == '__main__':
    main()