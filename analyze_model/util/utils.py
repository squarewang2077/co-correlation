import torch.nn as nn  

class Function_x(nn.Module):
    def __init__(self, x, u):
        super().__init__()
        self.x = x
        self.u = u
        self.c = x - u  

    def forward(self, t):
        t*self.u + self.c
        return t*self.u + self.c 
