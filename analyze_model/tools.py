import torch 
from torch import nn 
from tqdm import tqdm
# from functorch import jvp, vjp, vmap  
from torch import vmap 
from torch.func import jvp, vjp 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DictInNode:
    def __init__(self, dict_node: dict={}):
        self.dict_node = dict_node
        self.children = {} 
        self.parent = {}

    def __len__(self):
        return len(self.dict_node)

class nnTree(DictInNode, nn.Module):
    def __init__(self, dict_node):
        # super(nnTree, self).__init__(dict_node)
        DictInNode.__init__(self, dict_node)
        nn.Module.__init__(self)

        assert 'named_module' in self.dict_node, 'nnTree error: the node in module tree should at least contain the module and its name'
        assert hasattr(self.dict_node['named_module'][1], 'children'), 'nnTree error: the module has to have method .children'

        try:
            next(self.dict_node['named_module'][1].children())
        except StopIteration:
            self.depth = 0 # depth is 0 for tree with no children 
            return None 

        depth_subtree = [] # to store the depth of sub trees 
        sibling_index = 0
        for sub_module in self.dict_node['named_module'][1].children():
            sub_name = f'{sub_module.__class__.__name__}({sibling_index})'
            sub_dict_node = {}

            sub_dict_node['named_module'] = (sub_name, sub_module)
            for key in set(dict_node).difference({'named_module'}):
                self.dict_node[key].assign_child_params()
                sub_dict_node[key] = self.dict_node[key].add_child(name=sub_name)

            # add and initialize the children node, cannot define as nnTree() and add info in dict_node later,
            # since information in dict_node has to be initialized at first place 
            self.children[sub_name] = nnTree(sub_dict_node)  
            # recursively assign parent for each node 
            self.children[sub_name].parent[self.dict_node['named_module'][0]] = self # add parent

            depth_subtree.append(self.children[sub_name].depth) # append the depth of the sub trees                 
            sibling_index += 1 

        self.depth = 1 + max(depth_subtree) # the depth of parent tree is 1 + that of the child tree with maximum depth  

        # initialized when self.decompose is called
        # self.decomposed_named_modules = None 
        self.decomposed_modules = []

    def __str__(self) -> str:
        
        return self.dict_node['named_module'][1].__str__()

    def __repr__(self) -> str:
        return self.dict_node['named_module'][0]

    def is_leaf(self):
        if self.children == {}: 
            is_leaf = True
        else: 
            is_leaf = False
        return is_leaf

    def is_root(self):
        if self.parent == {}: 
            is_parent = True
        else: 
            is_parent = False
        return is_parent

    def reset(self):
        self.decomposed_modules = []

    def get_subtrees_of_level(self, level):
        # return of the function is trees 

        if level == 0 or self.children == {}: # for leaf and level 0 return module itself 
            # return [{self.dict_node['named_module'][0]: nnTree(self.dict_node)}]              
            # return [nnTree(self.dict_node)]
            return [self]

        modules = []
        current_level = level - 1
        for sub_module in self.children.values():
            modules += sub_module.get_subtrees_of_level(current_level)

        return modules 
        
    def decompose(self, decompose_route):
        # decompose the model into sub-modules
        # output: decomposed_modules

        # fetch the target trees 
        trees = self.get_subtrees_of_level(0) # starting from the root                         
        while len(decompose_route) > 0:
            new_trees = []
            for i, level in enumerate(decompose_route[0]):
                # new_trees += next(iter(trees[i].values())).get_subtrees_of_level(level)
                new_trees += trees[i].get_subtrees_of_level(level)
            trees = new_trees
            decompose_route.pop(0) # update the route until it's empty

        # capsule the tree in nn.sequence and give proper name for each module
        for tree in trees:
            # tree_value = next(iter(tree.values()))
            # self.decomposed_named_modules.append((tree_value.dict_node['named_module'][0], tree_value.dict_node['named_module'][1]))
            self.decomposed_modules.append(tree)

    def add_node(self):
        pass 

    def remove_node(self):
        pass 

    def forward(self, x):
        return self.dict_node['named_module'][1](x)


class RiskApproximator:
    def __init__(self, func, batched_x, device):
        self.func = func.to(device) 
        self.batched_x = batched_x.to(device) # batched_x is of (batch_size, ..., ..., ), e.g., (10, 1, 197, 512)
        self.device = device 

        # get after method has been conducted 
        self.msvs = None  
        self.F_norms = None
        self.svds = None 

    @staticmethod
    def get_aggr_dim(x):
        # the dimension index except 1th one e.g., for x = [10, 1, 197, 512]
        # aggr_dim_index = [1,2,3], 10 is corresponded to 0
        return [i for i in range(x.dim())][1:]

    @staticmethod
    def _F_norm_of_jacob(func, x):
            x.requires_grad_()
            y = func(x)
            inner_norms = []
            for i in tqdm(range(y.numel())):
                v = torch.zeros([y.numel()]).to(x.device)
                v[i] = 1
                v = v.reshape_as(y)
                vj = torch.autograd.grad(y, x, v, retain_graph=True)[0]
                inner_norms.append(torch.linalg.vector_norm(vj))
            inner_norms = torch.tensor(inner_norms)
            return torch.linalg.vector_norm(inner_norms)

    def F_norm_of_jacob(self):   
        self.F_norms = [self._F_norm_of_jacob(self.func, x).item() for x in tqdm(self.batched_x)]                            
            
    def msv_by_power_iteration(self, iters=10, tol=10e-5):

        batched_u = torch.rand_like(self.batched_x).to(self.device)
        batched_u /= torch.linalg.vector_norm(batched_u, dim=self.get_aggr_dim(batched_u)).view(-1, *(1,)*len(self.get_aggr_dim(batched_u)))
        batched_func = vmap(self.func)

        for i in range(iters):
            _, batched_v = jvp(batched_func, (self.batched_x,), (batched_u,))
            _, vjp_fn = vjp(batched_func, self.batched_x) 
            batched_u = vjp_fn(batched_v)[0]
            u_L2_norms = torch.linalg.vector_norm(batched_u, dim=self.get_aggr_dim(batched_u))
            v_L2_norms = torch.linalg.vector_norm(batched_v, dim=self.get_aggr_dim(batched_v))
            msvs = u_L2_norms/(v_L2_norms)
 
            # replace nan with zero when 0/0 happens
            msvs = torch.where(torch.isnan(msvs), torch.zeros_like(msvs), msvs)

            batched_u = batched_u/u_L2_norms.view(-1, *(1,)*len(self.get_aggr_dim(batched_u)))
            batched_v = batched_v/v_L2_norms.view(-1, *(1,)*len(self.get_aggr_dim(batched_v)))
            
            if self.msvs is not None:
                percent_error = (msvs - self.msvs)/(self.msvs)
                # replace nan with zero when 0/0 happens
                percent_error = torch.where(torch.isnan(percent_error), torch.zeros_like(percent_error), percent_error)
 
                # print(percent_error)
                if max(percent_error) < tol: break 
    
            self.msvs = msvs.detach()
        print(f'percent error: {percent_error}')

        self.msvs = self.msvs.tolist()

    def svd(self, top_k=1):
        self.svds = []
        for x in tqdm(self.batched_x):         
            jacob = torch.autograd.functional.jacobian(self.func, x)
            jacob_dim = x[0].shape[0]*x[0].shape[1]
            jacob = jacob.reshape([jacob_dim,jacob_dim])

            # calculate the singular value 
            svdvals = torch.linalg.svdvals(jacob)
            self.svds.append(torch.topk(svdvals, top_k)[0].tolist())

    def compute(self, method, **kwargs):
        if method == 'msv_by_power_iteration':
            self.msv_by_power_iteration(**kwargs)
        elif method == 'svd':
            self.svd(**kwargs)
        elif method == 'F_norm_of_jacobian':
            self.F_norm_of_jacob()
