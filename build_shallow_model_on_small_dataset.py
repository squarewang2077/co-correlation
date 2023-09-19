import torch
import math
import argparse 
import torch.nn as nn 


from build_model import *
from util import *
from build_model.util import *


def main():

    parser = argparse.ArgumentParser(description='train_attack_analysis_MLPs')

    # the class of the model, e.g., vit or covit  
    parser.add_argument('--net_type', type=str, default='mlp', choices=['mlp'], 
                        help='the type of the neural network, e.g., vit or covit')

    # configuration for shallow model
    parser.add_argument('--width', type=int, default=16, 
                        help='the width')
    parser.add_argument('--act_name', type=str, default='relu', 
                        help='the activation')
    parser.add_argument('--img_size', nargs='+', type=int, default=[1, 28, 28], 
                        help='the input image size, e.g., (224, 224), including channles')
    parser.add_argument('--n_classes', type=int, default=10, 
                        help='the number of classes')

    # dataset configuration 
    parser.add_argument('--dataset', type=str, default='minst', 
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
    parser.add_argument('--ckp_epochs', nargs='+', type=int, default=[i+1 for i in range(50)], 
                        help='the list to store the network')                        
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='show the training and evaluation details',)

    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the train loader, prepare for training 
    train_set, test_set = get_dataset(args, norm=True)
    net_name, net = get_network(args)
    net = net.to(DEVICE)    

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, net.parameters()), lr=0.003)
    lr_scheduler = None

    # initialization of the learner which is used to train and validate the network 
    learner = Learner((args.dataset, train_set, test_set), (net_name, net), optimizer, lr_scheduler, nn.CrossEntropyLoss(), device=DEVICE, \
                       ckp_folder = 'ckp/mlps/', result_path='results/mlps/training_results.csv')
    learner.train(args.start_epoch, args.end_epoch, args.batch_size, args.eval_freq, args.ckp_epochs, args.verbose)
 
if __name__ == '__main__':
    main()