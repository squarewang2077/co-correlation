from tqdm import tqdm

from util import *
from attack_model import * 
from analyze_model.util import *
from analyze_model.tools import *


class BaseAnalyser(Logger):
    def __init__(self, named_dataset, named_network, named_attack, analyze_depth, device, \
                 result_folder='results/'):
        super(BaseAnalyser, self).__init__()

        self.dataset_name, self.dataset = named_dataset
        self.net_name, self.net = named_network
        self.net = self.net.eval()
        self.atk_name, self.atk = named_attack
        self.analyze_depth = analyze_depth
        self.device = device
        self.result_folder = result_folder

        # initialized when decompose is called  
        self.module_tree = None 

        # initialized when subsample is called 
        self.dataloader = None 

    def save_to_log(self, result, method, **kwargs):        
        suffix = '_'.join([f'{k}({v})' for k, v in kwargs.items()])
        method = method.split('_')[0]  
        self._save_to_log(result, None, f'{self.result_folder}/depth-{self.analyze_depth}/{method}_{suffix}_for_{self.net_name}.csv')
        
    def decompose(self, decompose_route):
        # decompose model as basic modules 
        self.module_tree = nnTree({'named_module': (self.net_name, self.net)})
        self.module_tree.decompose(decompose_route)
    
    def subsample(self, frac_size, batch_size):
        self.dataloader = subsample(self.dataset, frac_size, batch_size)

    def _rinse(self):
        # initialize the result by giving a proper name for each module 
        # rename the module for better saving
        result = {}
        named_modules = {}
        module_index = 0
        for module in tqdm(self.module_tree.decomposed_modules):
            module_name = module.dict_node['named_module'][0] + '_' + str(module_index)
            named_modules[module_name] = module.dict_node['named_module'][1]
            result[module_name] = []
            module_index += 1

        result['label'] = []
        result['prediction'] = []
        # result['pred_adv'] = []

        return result, named_modules

    def stitch(self, module_name, module):

        if self.net_name[:3] == 'vit':
            # this is for the vits
            if module_name == 'PatchEmbed(0)_1':
                class ProstheticsPE(nn.Module):
                    def __init__(self, scalpels) -> None:
                        super().__init__()
                        self.scalpels = scalpels

                    def forward(self, x):
                        cls_token = self.scalpels.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                        if self.scalpels.dist_token is None:
                            x = torch.cat((cls_token, x), dim=1)
                        else:
                            x = torch.cat((cls_token, self.scalpels.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
                        x = self.scalpels.pos_drop(x + self.scalpels.pos_embed)                    
                        return x
                module = nn.Sequential(module, ProstheticsPE(self.net[1]))

            elif module_name[:9]  == 'Linear(5)':
                class ProstheticsLinear(nn.Module):
                    def __init__(self):
                        super().__init__()

                    def forward(self, x):
                        x = x[:,0]
                        return x

                module = nn.Sequential(ProstheticsLinear(), module)
        elif self.net_name[:5] == 'mixer':
            if module_name[:9]  == 'Linear(3)':
                class ProstheticsLinear(nn.Module):
                    def __init__(self):
                        super().__init__()

                    def forward(self, x):
                        x = x.mean(dim=1)
                        return x

                module = nn.Sequential(ProstheticsLinear(), module)

        return module 

    def analyze(self, saving, method, **kwargs):        

        result, named_modules = self._rinse()

        # filling the result 
        for imgs, labels in tqdm(self.dataloader):
            features, labels = imgs.to(self.device), labels.to(self.device)
            result['label'] += labels.tolist()
            # adv_imgs = self.atk(imgs, labels).to(self.device).detach() # implement the attack
            # result['pred_adv'] += self.net(adv_imgs).argmax(dim=1).detach().clone().tolist()
            
            for module_name, module in named_modules.items():
                # subsitude the nn.ReLU if is employed with inplace=True            
                if isinstance(module, nn.ReLU):
                    if module.inplace:
                        module = nn.ReLU(inplace=False)
                elif isinstance(module, nn.ReLU6):
                    if module.inplace:
                        module = nn.ReLU6(inplace=False)
                elif isinstance(module, nn.Hardswish):
                    if module.inplace:
                        module = nn.Hardswish(inplace=False)
                if self.net_name[:3] == 'vit':
                    if module_name == 'Dropout(1)_2':
                        result[module_name]+=[None]*len(features)
                        continue
                
                module = self.stitch(module_name, module)

                # change the additional batch size, since modules can only deal with size of (1, 197, 128) 
                features = torch.unsqueeze(features, 1) 

                risk = RiskApproximator(module, features, self.device)
                risk.compute(method, **kwargs)
                result[module_name] += risk.msvs

                features = module(features.squeeze(dim=1)) # the output of module_1 is the input of module_2

                # for test wheter the forward progation is correct!
                # (self.net(imgs.to(self.device))!=features).sum()

            result['prediction'] += features.argmax(dim=1).detach().clone().tolist()

        if saving:
            self.save_to_log(result, method, **kwargs)
     
        return result 
             
    def get_internal_distance(self, Lp_norm):

        dist_result, named_modules = self._rinse()

        for imgs, labels in tqdm(self.dataloader):

            features, labels = imgs.to(self.device), labels.to(self.device)
            adv_features = self.atk(features, labels).detach().clone().to(self.device) # implement the attack 

            dist_result['label'] += labels.tolist()

            # for distance for each input image
            for module_name, module in named_modules.items():                    

                # subsitude the nn.ReLU if is employed with inplace=True            
                if isinstance(module, nn.ReLU):
                    if module.inplace:
                        module = nn.ReLU(inplace=False)
                if self.net_name[:3] == 'vit':
                    if module_name == 'Dropout(1)_2':
                        dist_result[module_name]+=[None]*len(features)
                        continue
                module = self.stitch(module_name, module)

                features = module(features)
                adv_features = module(adv_features)
                dist_result[module_name] += dist_of(features, adv_features, Lp_norm).tolist()

            dist_result['prediction'] += features.argmax(dim=1).detach().clone().tolist()
            dist_result['pred_adv'] += adv_features.argmax(dim=1).detach().clone().tolist()
        self._save_to_log(dist_result, None, f'results/{self.atk_name}_for_{self.net_name}.csv')
 