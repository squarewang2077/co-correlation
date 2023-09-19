import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import math
from functools import partial, reduce
import operator
import timm 


class cnn(nn.Module):
    def __init__(self, img_size, kernel_size, n_classes):
        super().__init__()
        # defination of variable
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.n_cls = n_classes
        
        self.padding = (kernel_size - 1)/2
        # defination of the functions 
        self.layer1 = nn.Sequential(
                                    nn.Conv2d(in_channels=3, out_channels=128, kernel_size=kernel_size, stride=2, padding=self.padding), \
                                    nn.ReLU()
                                    )         
        self.head = nn.Linear(in_features=reduce(operator.mul, img_size, 1)/2, out_features=n_classes, bias=False)        
        # initialization of the paramets
        self._init_parameters(0.5)

        # freeze the last layer 
        for p in self.head.parameters():
            p.requires_grad_(False)


    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.head(x)
        return x 

    def _init_parameters(self, p):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1/math.sqrt(self.width**(1+p)))
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.constant_(m.bias, 0)

# class WRN50_rpc(nn.Module):
#     def __init__(self, pretrained, n_cls, partitiion_index, init_par):
#         super().__init__()
#         resnet50 = timm.create_model('wide_resnet50_2', pretrained)
#         resnet50.fc = nn.Linear(resnet50.fc.weight.shape[1], n_cls)
#         # freeze the last layer 
#         for p in resnet50.fc.parameters():
#             p.requires_grad_(False)

#         resnet = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.act1, resnet50.maxpool, \
#                       resnet50.layer1, resnet50.layer2, resnet50.layer3, resnet50.layer4, resnet50.global_pool, resnet50.fc)
#         self.layer1 = resnet[:partitiion_index]
#         self.layer2 = resnet[partitiion_index:]

#         # weight initalization 
#         # self._init_parameters(init_par)

#     def forward(self, x):
#         x = self.layer1(x) 
#         x = self.layer2(x)
#         return x

#     def _init_parameters(self, p):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, 0, 0, 1/math.sqrt(m.weight.shape[0]**(1+p)))


# class WRN50_rpc(nn.Module):
#     def __init__(self, pretrained, n_cls, partitiion_index, init_par):
#         super().__init__()
#         resnet50 = timm.create_model('wide_resnet50_2', pretrained)
#         resnet50.fc = nn.Linear(resnet50.fc.weight.shape[1], n_cls)
#         # freeze the last layer 
#         for p in resnet50.fc.parameters():
#             p.requires_grad_(False)

#         resnet = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.act1, resnet50.maxpool, \
#                       resnet50.layer1, resnet50.layer2, resnet50.layer3, resnet50.layer4, resnet50.global_pool, resnet50.fc)
#         self.layer1 = resnet[:partitiion_index]
#         self.layer2 = resnet[partitiion_index:]

#         # weight initalization 
#         # self._init_parameters(init_par)

#     def forward(self, x):
#         x = self.layer1(x) 
#         x = self.layer2(x)
#         return x

#     def _init_parameters(self, p):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, 0, 0, 1/math.sqrt(m.weight.shape[0]**(1+p)))


# class resnet50_rpc(nn.Module):
#     def __init__(self, pretrained, n_cls, partitiion_index, init_par):
#         super().__init__()
#         resnet50 = timm.create_model('resnet50', pretrained)
#         resnet50.fc = nn.Linear(resnet50.fc.weight.shape[1], n_cls)
#         # freeze the last layer 
#         for p in resnet50.fc.parameters():
#             p.requires_grad_(False)

#         resnet = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.act1, resnet50.maxpool, \
#                       resnet50.layer1, resnet50.layer2, resnet50.layer3, resnet50.layer4, resnet50.global_pool, resnet50.fc)
#         self.layer1 = resnet[:partitiion_index]
#         self.layer2 = resnet[partitiion_index:]

#         # weight initalization 
#         # self._init_parameters(init_par)

#     def forward(self, x):
#         x = self.layer1(x) 
#         x = self.layer2(x)
#         return x

#     def _init_parameters(self, p):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, 0, 0, 1/math.sqrt(m.weight.shape[0]**(1+p)))

class resnet50_rpc(nn.Module):
    def __init__(self, iswide, ispretrained, n_cls, init_par):
        super().__init__()
        # choise of the model
        if iswide:
            resnet50 = timm.create_model('wide_resnet50_2', ispretrained)    
        else:
            resnet50 = timm.create_model('resnet50', ispretrained)

        # replace the last layer
        resnet50.fc = nn.Linear(resnet50.fc.weight.shape[1], n_cls)

        # freeze the last layer 
        for p in resnet50.fc.parameters():
            p.requires_grad_(False)

        # repack the model
        self.embedding = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.act1, resnet50.maxpool) 

        body_1 = nn.Sequential(resnet50.layer1, resnet50.layer2)
        body_2 = nn.Sequential(resnet50.layer3, resnet50.layer4)
        self.body = nn.Sequential(body_1, body_2)

        self.cls_head = nn.Sequential(resnet50.global_pool, resnet50.fc)

        # weight initalization 
        # self._init_parameters(init_par)

    def forward(self, x):
        x = self.embedding(x) 
        x = self.body(x)
        x = self.cls_head(x)
        return x

    def _init_parameters(self, p):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0, 1/math.sqrt(m.weight.shape[0]**(1+p)))


class vgg11_rpc(nn.Module):
    def __init__(self, pretrained, n_cls, init_par):
        super().__init__()
        vgg11 = timm.create_model('vgg11', pretrained)
        vgg11.head.fc = nn.Linear(4096, n_cls)

        # freeze the last layer 
        for p in vgg11.head.parameters():
            p.requires_grad_(False)

        # repackage
        self.layer1 = vgg11.features
        self.layer2 = nn.Sequential(vgg11.pre_logits, vgg11.head)

        # initalization 
        # self._init_parameters(init_par)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def _init_parameters(self, p):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1/math.sqrt(m.weight.shape[0]**(1+p)))