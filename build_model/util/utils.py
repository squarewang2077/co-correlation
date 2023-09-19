import re
import os
import math
import torch
from PIL import Image
import itertools
from torchvision import transforms, datasets
from build_model.model_zoo import * 
from build_model.optimizers import * 

def get_imagenet(net_cft_dict, root = './dataset/imagenet', train = False):
    crop_pct = net_cft_dict['crop_pct']
    img_size = 224    
    scale_size =  int(math.floor(img_size / crop_pct))
    if net_cft_dict['interpolation'] == 'bicubic':
        interpolationmode = transforms.InterpolationMode.BICUBIC
    else :
        interpolationmode = transforms.InterpolationMode.BILINEAR
    
    transform = transforms.Compose([
                                    transforms.Resize(scale_size, interpolation=interpolationmode),
                                    transforms.CenterCrop(img_size),
                                    transforms.ToTensor(),
                                    ])
    target_transform = None

    if train:
        root = os.path.join(root, 'train')
    else:
        root = os.path.join(root, 'val')

    return datasets.ImageFolder(root = root,
                               transform = transform,
                               target_transform = target_transform)

def get_dataset(args, norm=False, aug=False, resize=False):
    train_transform_control = [c for c in [aug, resize, True, norm]]
    test_transform_control = [c for c in [False, resize, True, norm]]
    # transformation for training set
    _transforms = []
    _transforms.append([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    _transforms.append([
        transforms.Resize(args.transform_resize)
    ])

    _transforms.append([
        transforms.ToTensor()
    ])

    if args.dataset == 'minst':
        _transforms.append([
            transforms.Normalize(mean=(0.5,), std=(0.5,))
            ])    
    else:
        _transforms.append([
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])    


    transform_train = transforms.Compose([t[0] for t, c in zip(_transforms, train_transform_control) if c])
    transform_test = transforms.Compose([t[0] for t, c in zip(_transforms, test_transform_control) if c])

    saving_path = './dataset'

    if args.dataset == 'minst':
        train_set = datasets.MNIST(root=saving_path,
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        test_set = datasets.MNIST(root=saving_path,
                                   train=False,
                                   download=True,
                                   transform=transform_test) 

    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(root=saving_path,
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        test_set = datasets.CIFAR10(root=saving_path,
                                   train=False,
                                   download=True,
                                   transform=transform_test) 


    elif args.dataset == "cifar100":
        train_set = datasets.CIFAR100(root=saving_path,
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        test_set = datasets.CIFAR100(root=saving_path,
                                    train=False,
                                    download=True,
                                    transform=transform_test) 
    elif args.dataset == "svhn":
        train_set = datasets.SVHN(root=saving_path,
                                     split='train',
                                     download=True,
                                     transform=transform_train)

        test_set = datasets.SVHN(root=saving_path,
                                    split='test',
                                    download=True,
                                    transform=transform_test) 

    return train_set, test_set

def get_network(args):
    if args.net_type == "vit":
        # the dict conf is for debugging 
        conf={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
            'embedding':{
                            'in_channels': args.in_channels,
                            'img_size': args.img_size,
                            'patch_size': args.patch_size,
                            'em_size': args.em_size
        }, 
                'encoder':{
                            'depth': args.depth,
                            'd_K': args.d_K,
                            'd_V':args.d_V, 
                            'num_heads': args.num_heads, # d_K = heads * d_k
                            'att_drop_out': args.att_drop_out,

                            'MLP_expansion': args.MLP_expansion,
                            'MLP_drop_out': args.MLP_drop_out 

        },
                'cls_head':{
                            'n_classes': args.n_classes
        }
        }

        net = ViT(
            in_channels=conf['embedding']['in_channels'],
            patch_size=conf['embedding']['patch_size'],
            em_size=conf['embedding']['em_size'],
            img_size=conf['embedding']['img_size'],
            depth=conf['encoder']['depth'],
            n_classes=conf['cls_head']['n_classes'],
            
            forward_expansion = conf['encoder']['MLP_expansion'],
            forward_drop_out = conf['encoder']['MLP_drop_out'],
            d_K = conf['encoder']['d_K'],
            d_V = conf['encoder']['d_V'],
            num_heads = conf['encoder']['num_heads'],
            drop_out = conf['encoder']['att_drop_out']
            ) 

        # name the net work: e.g., vit_D4_E512_H1_P16
        net_name = f'{args.net_type}_D{args.depth}_E{args.em_size}_H{args.num_heads}_P{args.patch_size[0]}'

    elif args.net_type == "covit":
        conf = { # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': args.in_channels,
                        'img_size': args.img_size,
                        'patch_size': args.patch_size,
                        'em_size': args.em_size
        }, 
            'encoder':{
                        'depth': args.depth,
                        'kernel_size_group': args.kernel_size_group, # for Conv1D 
                        'stride_group': args.stride_group,
                        'padding_group': args.padding_group,

                        'MLP_expansion': args.MLP_expansion,
                        'MLP_drop_out': args.MLP_drop_out 

        },
            'cls_head':{
                        'n_classes': args.n_classes
        }
        }

        net = CoViT(
            in_channels=conf['embedding']['in_channels'],
            patch_size=conf['embedding']['patch_size'],
            em_size=conf['embedding']['em_size'],
            img_size=conf['embedding']['img_size'],
            depth=conf['encoder']['depth'],
            n_classes=conf['cls_head']['n_classes'],
            
            forward_expansion = conf['encoder']['MLP_expansion'],
            forward_drop_out = conf['encoder']['MLP_drop_out'],
            kernel_size_group = conf['encoder']['kernel_size_group'],
            stride_group = conf['encoder']['stride_group'],
            padding_group = conf['encoder']['padding_group'],
            ) 

        # name the net work: e.g., covit_D1_E512_K3_P16
        kernel_size_group = [str(i) for i in args.kernel_size_group]
        kernel_size_group = ''.join(kernel_size_group)
        net_name = f'{args.net_type}_D{args.depth}_E{args.em_size}_H{kernel_size_group}_P{args.patch_size[0]}'
    elif args.net_type == 'mlp':
        net = shallow_model(args.img_size, args.width, args.act_name, args.n_classes)
        net_name = f'{args.net_type}_A({args.act_name})_W({args.width})'
    elif args.net_type == 'twolayer':
        net = twolayer_model(args.img_size, args.width_1, args.width_2, args.act_name, args.n_classes)
        net_name = f'{args.net_type}_A({args.act_name})_W({args.width_1}-{args.width_2})'
    elif args.net_type == 'vgg11':
        net = vgg11_rpc(args.pretrain, args.n_cls, args.init_par)
        net_name = f'{args.net_type}_pretrain({args.pretrain})_init_par({args.init_par})'
    elif args.net_type == 'resnet50':
        net = resnet50_rpc(False, args.pretrain, args.n_cls, args.init_par)
        net_name = f'{args.net_type}_pretrain({args.pretrain})_init_par({args.init_par})'
    elif args.net_type == 'wrn50':
        net = resnet50_rpc(True, args.pretrain, args.n_cls, args.init_par)
        net_name = f'{args.net_type}_pretrain({args.pretrain})_init_par({args.init_par})'


    return net_name, net 

def net_name_analyzer(detailed_net_name): 
    config = {}
    net_name_list = detailed_net_name.split('_')
    config['net_type'] = net_name_list[0]
    config['depth'] = int(net_name_list[1][1:])
    config['em_size'] = int(net_name_list[2][1:])
    if config['net_type'] == 'vit':
        config['num_heads'] = int(net_name_list[3][1:])
    elif config['net_type'] == 'covit':
        config['kernel_size_group'] = [int(x) for x in net_name_list[3][1:]]
        config['stride_group'] = [ 1 for k in config['kernel_size_group']] 
        config['padding_group'] = [int((k-s)/2) for k, s in zip(config['kernel_size_group'], config['stride_group'])]
    config['patch_size'] = (int(re.search(r'\w\d+', net_name_list[4]).group()[1:]), int(re.search(r'\w\d+', net_name_list[4]).group()[1:]))
    config['epoch'] = int(re.search(r'\(\d+\)', net_name_list[4]).group()[1:-1]) 

    return config

def get_network_by_name(args):
    
    config = net_name_analyzer(args.detailed_net_name)
    if args.dataset == 'cifar10':
       config['in_channels'] = 3 
       config['img_size'] = (args.transform_resize, args.transform_resize)
       config['n_classes'] = 10         

    if config['net_type'] == "vit":
        # the dict conf is for debugging 
        net = ViT(
            in_channels=config['in_channels'],
            patch_size=config['patch_size'],
            em_size=config['em_size'],
            img_size=config['img_size'],
            depth=config['depth'],
            n_classes=config['n_classes'],
            
            forward_expansion = 4,
            forward_drop_out = 0.,
            d_K = config['em_size'],
            d_V = config['em_size'],
            num_heads = config['num_heads'],
            drop_out = 0.
            ) 

    elif config['net_type'] == "covit":
        net = CoViT(
            in_channels=config['in_channels'],
            patch_size=config['patch_size'],
            em_size=config['em_size'],
            img_size=config['img_size'],
            depth=config['depth'],
            n_classes=config['n_classes'],
            
            forward_expansion = 4,
            forward_drop_out = 0.,
            kernel_size_group = config['kernel_size_group'],
            stride_group = config['stride_group'],
            padding_group = config['padding_group'],
            ) 

    return net     

def get_opt(args, net):
    if args.opt_name == 'sam':
        base_optimizer =torch.optim.SGD
        optimizer = SAM(net.parameters(), base_optimizer, lr=args.lr, momentum=args.momentum)    
    elif args.opt_name == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    elif args.opt_name == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    return optimizer

def get_lr_scheduler(args, opt, **kwargs):
    lr_sche = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=int(args.end_epoch - args.start_epoch) + 1, **kwargs)
    return lr_sche


def weight_init(net):
    pass
    return net 