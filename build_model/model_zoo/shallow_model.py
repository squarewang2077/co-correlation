import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import math
from functools import partial, reduce
import operator


# Use reduce to multiply all the elements


# class LinearEmbedding(nn.Sequential):
#     def __init__(self, img_size, expansion):
#         super().__init__(
#             nn.Flatten(),
#             nn.Linear(in_features=torch.tensor(img_size).prod().item(), out_features=int((torch.tensor(img_size).prod()*expansion).item()))
#         )


# class ClassificationHead(nn.Sequential):
#     def __init__(self, input_size, n_class):
#         super().__init__(
#             nn.Linear(in_features=input_size, out_features=n_class)
#         )

# class MLPBlock(nn.Sequential):
#     def __init__(self, **kwargs):
#         super().__init__(
#             nn.Linear(**kwargs),
#             nn.ReLU()        
#         )


# class MLP(nn.Module):
#     def __init__(self, img_size, expansion, n_class, depth):
#         super().__init__()
#         self.depth = depth

#         width_func = partial(self.width_func, img_size = img_size, expansion = expansion)
#         self.embedding = LinearEmbedding(img_size, expansion)
#         self.blocks = nn.Sequential(*[MLPBlock(in_features = width_func(i)[0], out_features = width_func(i)[1]) for i in range(depth)]) 
#         self.classificationhead = ClassificationHead(width_func(self.depth - 1)[1], n_class) 

#     @staticmethod
#     def width_func(i, img_size, expansion):
        
#         input_size = int((torch.tensor(img_size).prod()*expansion).item()) 
#         output_size = input_size

#         return (input_size, output_size)
                
#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.blocks(x)
#         x = self.classificationhead(x)

#         return x 


class shallow_model(nn.Module):
    def __init__(self, img_size, width, act_name, n_classes):
        super().__init__()
        # defination of variable
        self.img_size = img_size
        self.width = width
        self.n_cls = n_classes
        
        self.flatten = nn.Flatten()

        # defination of the functions 
        if act_name == 'relu':
            self.layer1 = nn.Sequential(
                                        nn.Linear(in_features=reduce(operator.mul, img_size, 1), out_features=width, bias=False), \
                                        nn.ReLU()
                                        )         
        elif act_name == 'linear':
            self.layer1 = nn.Sequential(
                                        nn.Linear(in_features=reduce(operator.mul, img_size, 1), out_features=width, bias=False), \
                                        nn.Identity()
                                        ) 

        self.head = nn.Linear(in_features=width, out_features=n_classes, bias=False)        
        # initialization of the paramets
        self._init_parameters(-0.3)

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



class twolayer_model(nn.Module):
    def __init__(self, img_size, width_1, width_2, act_name, n_classes):
        super().__init__()
        # defination of variable
        self.img_size = img_size
        self.width_1 = width_1
        self.width_2 = width_2
        self.n_cls = n_classes
        
        self.flatten = nn.Flatten()

        # defination of the functions 
        if act_name == 'relu':
            self.layer_1 = nn.Sequential(
                                        nn.Linear(in_features=reduce(operator.mul, img_size, 1), out_features=width_1), \
                                        nn.ReLU()
                                        )         
        elif act_name == 'linear':
            self.layer_1 = nn.Sequential(
                                        nn.Linear(in_features=reduce(operator.mul, img_size, 1), out_features=width_1), \
                                        nn.Identity()
                                        ) 

        self.layer_2 = nn.Linear(in_features=width_1, out_features=width_2)

        self.head = nn.Linear(in_features=width_2, out_features=n_classes)        
        # initialization of the paramets for sub-modules
        self._init_parameters()    

        # freeze the last layer 
        for p in self.head.parameters():
            p.requires_grad_(False)


    def forward(self, x):
        x = self.flatten(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.head(x)
        return x 

    def _init_parameters(self):
        for m in self.layer_1.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1/math.sqrt(self.width_1**(1+0.5)))
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        for m in self.layer_2.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1/math.sqrt(self.width_2**(1+0.5)))
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

